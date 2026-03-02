import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Mapping
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from collections import defaultdict


from utils.log_utils import configure_cache_logger
from blueprint_generation import generate_new_blueprint


@dataclass
class AgentBlueprint:
    tag: str
    type: str
    steps: List[str]
    tool_signature: Dict[str, bool]


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
        cache_enabled: When False, blueprints are never stored (baseline mode).
        blueprint_db: In-memory list of stored blueprints.
        vector_index: In-memory list of embedding vectors aligned with
            ``blueprint_db``.
    """
    def __init__(self, cache_enabled: bool = True):
        self.ner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_enabled = cache_enabled
        self.blueprint_db: List[AgentBlueprint] = []
        self.vector_index = []
        _dt = datetime.now().strftime("%m%d_%H%M")
        self.hit_logger = configure_cache_logger("cache_hits", f"cache_hits_{_dt}.log")
        self.miss_logger = configure_cache_logger("cache_misses", f"cache_misses_{_dt}.log")
        # Internal data trackers for performance evaluation
        self._similarity_scores = []
        self._cache_hits = 0
        self._query_count = 0

        self.HIT_THRESHOLD = 0.9
    
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
        masked_query, variables, tool_signature = self._extract_and_mask(user_query)
        task_type = self._build_task_prefix(user_query, tool_signature)

        # Key query for vector comparison
        key_query = f"{task_type} {masked_query}"

        # Instantly generate + return if cache empty
        if not self.vector_index:
            self.miss_logger.info("EMPTY_DB_MISS | key_query=%s", key_query)
            blueprint, inp_tokens, out_tokens =  self._gen_blueprint_to_db(masked_query, key_query, task_type, tool_signature=tool_signature)
            return {
                "tag": key_query,
                "blueprint": blueprint,
                "inp_tokens": inp_tokens,
                "out_tokens": out_tokens,
                "variables": variables,
                "score": 0.0,
                "status": "MISS"
            }

        # Convert from tensor
        key_vec = self.embedder.encode(key_query)
        if hasattr(key_vec, "detach"):  # torch tensor
            key_vec = key_vec.detach().cpu().numpy()
        key_vec = np.asarray(key_vec, dtype=np.float32).reshape(1, -1)

        valid_indices = [i for i, bp in enumerate(self.blueprint_db) if bp.type == task_type]
        
        if not valid_indices:
            self.miss_logger.info("TYPE_MISS | type=%s | key_query=%s", task_type, key_query)
            blueprint, inp_tokens, out_tokens = self._gen_blueprint_to_db(masked_query, key_query, task_type, tool_signature)
            return {
                "tag": key_query,
                "blueprint": blueprint,
                "inp_tokens": inp_tokens,
                "out_tokens": out_tokens,
                "variables": variables,
                "score": 0.0,
                "status": "MISS"
            }
        # Extract closest match from the FILTERED vector space
        valid_vecs = np.array([self.vector_index[i] for i in valid_indices])
        scores = cosine_similarity(key_vec, valid_vecs)[0]
        best_local_idx = np.argmax(scores)
        best_idx = valid_indices[best_local_idx]
        
        matched_blueprint = self.blueprint_db[best_idx]
        max_score = scores[best_local_idx]
        self._similarity_scores.append(max_score)

        # Default 0 if cache hit
        inp_tokens = 0
        out_tokens = 0
    
        # Cache miss check
        if max_score < self.HIT_THRESHOLD:
            self.miss_logger.info(
                "THRESHOLD_MISS | key_query=%s | best_match_tag=%s | score=%.6f | threshold=%.2f",
                key_query,
                matched_blueprint.tag,
                float(max_score),
                self.HIT_THRESHOLD,
            )
            matched_blueprint, inp_tokens, out_tokens = self._gen_blueprint_to_db(masked_query,key_query, task_type,tool_signature)
            final_status = "MISS"
        else:
            self._cache_hits += 1
            self.hit_logger.info(
                "CACHE_HIT | key_query=%s | matched_tag=%s | score=%.6f",
                key_query,
                matched_blueprint.tag,
                float(max_score),
            )
            final_status = "HIT"

        return {
                "tag": key_query,
                "blueprint": matched_blueprint,
                "inp_tokens": inp_tokens,
                "out_tokens": out_tokens,
                "variables": variables,
                "score": float(max_score),
                "status": final_status     
            }


    def add_blueprint(self, blueprint: AgentBlueprint) -> None:
        """Add a blueprint and cache its embedding.

        Args:
            blueprint: Blueprint object to store.

        Returns:
            None
        """
        if not self.cache_enabled:
            return
        vector = self.embedder.encode(blueprint.tag)
        self.blueprint_db.append(blueprint)
        self.vector_index.append(vector)

    
    def _build_task_prefix(self, query: str, tool_signature: Dict[str, bool]) -> str:
        """Zero-latency heuristic to classify the expected output type."""
        query_lower = query.lower()
        
        prefix_parts = []

        # Keywords that strongly indicate text analysis
        explanation_keywords = ['why', 'drove', 'explain', 'reason', 'factors', 'impact', 'how did', 'cause']
        comparison_keywords = ["higher", "lower", "larger", "smaller", "largest", "smallest", "lowest", "highest"]


        if any(kw in query_lower for kw in explanation_keywords):
            prefix_parts.append("[EXPLANATION]")
        elif any(kw in query_lower for kw in comparison_keywords):
            prefix_parts.append("[COMPARISON]")

        if tool_signature["needs_math"]:
                    if "defined as:" in query_lower:
                        prefix_parts.append("[MATH: CUSTOM_FORMULA]")

                    elif "year-over-year" in query_lower or "yoy" in query_lower or "change" in query_lower:
                        prefix_parts.append("[MATH: YOY_CHANGE]")

                    elif "average" in query_lower:
                        prefix_parts.append("[MATH: AVERAGE]")

                    elif "ratio" in query_lower or "margin" in query_lower:
                        prefix_parts.append("[MATH: RATIO]")

                    else:
                        prefix_parts.append("[MATH: GENERAL]")

        if not prefix_parts:
            prefix_parts.append("[EXTRACTION]")

        return " ".join(prefix_parts)

    def _extract_and_mask(self, query: str) -> Tuple[str, Mapping[str, str], Mapping[str, bool]]:
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
        variables = defaultdict(list)
        masked_query = query.lower() 


        # TODO: figure out 20-30 optimal labels for extraction, need to be representative for all financebench
        labels = ["financial metric",  "unit of financial quantity", 
                  "financial data document", "metric", "company", "year",
                  "mathematical operation"] 
        entities = self.ner.predict_entities(query, labels, threshold=0.3)

        tool_signature = {
            "needs_math": False
        }
        
        for e in entities:
            label = e['label']
            text = e['text'].lower()

            if label == "mathematical operation":
                tool_signature["needs_math"] = True

            variables[label].append(text)
            masked_query = masked_query.replace(text, f"[{label}]")
            

        return masked_query, variables, tool_signature 
    
    def _gen_blueprint_to_db(self, masked_query: str, key: str, task_type: str, tool_signature: Dict[str, bool]) -> tuple[AgentBlueprint, int, int]:
        """
        Return a fully reasoned blueprint object for a given masked query.

        Uses Vertex AI api for gemini 2.5 pro to perform heavy reasoning if a blueprint is not found for a given prompt.

        Arguments:
            masked_query: generic query for which blueprint will be generated
            key: the final key query we use as a tag in the cache

        Returns:
            tuple[AgentBlueprint, int, int]
        """
        raw_json, in_tokens, out_tokens = generate_new_blueprint(masked_query)
        blueprint = json.loads(raw_json)
            
        agent_blueprint = AgentBlueprint(tag=key, type = task_type, steps=blueprint["steps"], tool_signature=tool_signature)
        self.add_blueprint(agent_blueprint)

        return agent_blueprint, in_tokens, out_tokens



        




