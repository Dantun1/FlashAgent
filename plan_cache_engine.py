import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


from utils.log_utils import configure_cache_logger
from blueprint_generation import generate_new_blueprint


@dataclass
class AgentBlueprint:
    tag: str
    steps: List[str]


class PlanCacheEngine:
    """
    Vector-based blueprint retrieval engine.

    This class stores blueprint descriptions as embeddings and retrieves
    the closest matching blueprint for a user query after masking
    extracted entities.

    (In future, we will integrate a prod-ready cache store like Redis)

    Attributes:
        ner: GLiNER model used for entity extraction.
        embedder: Sentence transformer used to generate embeddings.
        blueprint_db: In-memory list of stored blueprints.
        vector_index: In-memory list of embedding vectors aligned with
            ``blueprint_db``.
    """
    def __init__(self):
        self.ner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.blueprint_db: List[AgentBlueprint] = []
        self.vector_index = []
        self.hit_logger = configure_cache_logger("cache_hits", "cache_hits.log")
        self.miss_logger = configure_cache_logger("cache_misses", "cache_misses.log")
        # Internal data trackers for performance evaluation
        self._similarity_scores = []
        self._cache_hits = 0
        self._query_count = 0

        self.HIT_THRESHOLD = 0.8
    
    @property
    def similarities(self):
        return list(self._similarity_scores)
    
    @property
    def hit_stats(self):
        return (self._cache_hits, self._query_count, self._cache_hits / self._query_count)

    def retrieve_plan(self, user_query: str) -> Dict[str, Any]:
        """Retrieve the best matching blueprint.

        This will be used to create the final prompt

        Args:
            user_query: Raw user query.

        Returns:
            dict with:
                - ``tag``: the final key query that was used in cache lookup
                - ``blueprint``: the generic blueprint object
                - ``variables``: extracted entities
        """
        self._query_count += 1
        
        # Raw masked query
        masked_query, variables = self._extract_and_mask(user_query)
        task_type = self._get_task_type(user_query)

        # Key query for vector comparison
        key_query = f"{task_type} {masked_query}"

        # Instantly generate + return if cache empty
        if not self.vector_index:
            self.miss_logger.info("EMPTY_DB_MISS | key_query=%s", key_query)
            blueprint =  self._gen_blueprint_to_db(masked_query, key_query)
            return {
                "tag": key_query,
                "blueprint": blueprint,
                "variables": variables
                }

        # Convert from tensor
        key_vec = self.embedder.encode(key_query)
        if hasattr(key_vec, "detach"):  # torch tensor
            key_vec = key_vec.detach().cpu().numpy()
        key_vec = np.asarray(key_vec, dtype=np.float32).reshape(1, -1)

        # Extract closest match blueprint with cosine similarity
        db_vecs = np.array(self.vector_index)
        scores = cosine_similarity(key_vec, db_vecs)[0]
        best_idx = np.argmax(scores)
        
        matched_blueprint = self.blueprint_db[best_idx]
        max_score = scores[best_idx]
        self._similarity_scores.append(max_score)
    
        # Cache miss check
        if max_score < self.HIT_THRESHOLD:
            self.miss_logger.info(
                "THRESHOLD_MISS | key_query=%s | best_match_tag=%s | score=%.6f | threshold=%.2f",
                key_query,
                matched_blueprint.tag,
                float(max_score),
                self.HIT_THRESHOLD,
            )
            matched_blueprint = self._gen_blueprint_to_db(masked_query,key_query)
        else:
            self._cache_hits += 1
            self.hit_logger.info(
                "CACHE_HIT | key_query=%s | matched_tag=%s | score=%.6f",
                key_query,
                matched_blueprint.tag,
                float(max_score),
            )

        return {
            "tag": key_query,
            "blueprint" : matched_blueprint,
            "variables": variables,
        }


    def add_blueprint(self, blueprint: AgentBlueprint) -> None:
        """Add a blueprint and cache its embedding.

        Args:
            blueprint: Blueprint object to store.

        Returns:
            None
        """
        vector = self.embedder.encode(blueprint.tag)
        self.blueprint_db.append(blueprint)
        self.vector_index.append(vector)

    
    def _get_task_type(self, query: str) -> str:
        """Zero-latency heuristic to classify the expected output type."""
        query_lower = query.lower()
        
        # Keywords that strongly indicate text analysis
        explanation_keywords = ['why', 'drove', 'explain', 'reason', 'factors', 'impact', 'how did', 'cause']
        
        if any(kw in query_lower for kw in explanation_keywords):
            return "[EXPLANATION]"
        elif "compare" in query_lower or "higher" in query_lower or "lower" in query_lower:
            return "[COMPARISON]"
        else:
            return "[EXTRACTION]"


    def _extract_and_mask(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Given a query, mask out the variables, return the masked query and variables dict.

        Args:
            query: Raw user query string.

        Returns:
            Tuple containing:
                - masked query string
                - mapping from entity label to extracted text

        Uses GLiNER entity recognition.
        """
        variables = {}
        masked_query = query.lower() 


        # TODO: figure out 20-30 optimal labels for extraction, need to be representative for all financebench
        labels = ["financial metric",  "unit of financial quantity", "financial data document", "metric", "company", "year"] 
        entities = self.ner.predict_entities(query, labels, threshold=0.3)
        
        for e in entities:
            variables[e['label']] = e['text']
            # not performant, just direct string replace for tsting
            masked_query = masked_query.replace(e['text'].lower(), f"[{e['label']}]")

        return masked_query, variables
    
    def _gen_blueprint_to_db(self, masked_query: str, key: str) -> AgentBlueprint:
        """
        Return a fully reasoned blueprint object for a given masked query.

        Uses Vertex AI api for gemini 2.5 pro to perform heavy reasoning if a blueprint is not found for a given prompt.

        Arguments:
            masked_query: generic query for which blueprint will be generated
            key: the final key query we use as a tag in the cache

        Returns:
            AgentBlueprint
        """
        blueprint = json.loads(generate_new_blueprint(masked_query))
            
        # Add to DB 
        agent_blueprint = AgentBlueprint(
                tag = key,
                steps = blueprint["steps"],
            )
        self.add_blueprint(agent_blueprint)

        return agent_blueprint



        




