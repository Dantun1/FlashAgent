import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from gliner import GLiNER
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

from blueprint_generation import run_vertex_teacher

# @dataclass
# class BlueprintStep:
#     step_number: int
#     required_variables: List[str] 
    
#     def render(self, extracted_vars: dict) -> str:
#             step_vars = {k: extracted_vars.get(k) for k in self.required_variables}
#             return f"Executing: STEP {self.step_number}\nVariables: {step_vars}"

@dataclass
class AgentBlueprint:
    tag: str 
    steps: List[str]

class PlanCacheEngine:
    """
    Vector-based blueprint retrieval engine.

    This class stores blueprint descriptions as embeddings and retrieves the
    closest matching blueprint for a user query after masking extracted entities.

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
        # vectors not hashable, store as parallel arrays instead of dict.
        self.vector_index = []

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
        print(f"Stored Blueprint: [{blueprint.tag}]\n Stored steps: {blueprint.steps}")

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
    


    def retrieve_plan(self, user_query: str) -> Dict[str, Any] | str:
        """Retrieve the best matching blueprint.

        This will be used to create the final prompt

        Args:
            user_query: Raw user query.

        Returns:
            ``"DB Empty"`` when no blueprints are stored, otherwise a dict with:
                - ``matched_id``: matched blueprint ID
                - ``masked_intent``: masked query used for retrieval
                - ``blueprint``: the generic blueprint object
                - ``variables``: extracted entities
                
        """

        masked_query, variables = self._extract_and_mask(user_query)
        print(f"masked query: {masked_query} , variables: {variables}")

        query_vec = self.embedder.encode(masked_query).reshape(1, -1)
        
        if not self.vector_index:
            return "DB Empty"
            
        db_vecs = np.array(self.vector_index)
        scores = cosine_similarity(query_vec, db_vecs)[0]
        best_idx = np.argmax(scores)

        matched_blueprint = self.blueprint_db[best_idx]
        max_score = scores[best_idx]
    
        if max_score < 0.7:
            print(f"Cache Miss: {max_score} < 0.7, Generating blueprint")
            blueprint = json.loads(run_vertex_teacher(masked_query))
            
            # add agentblueprint to db, tag is the masked_query
            agent_blueprint = AgentBlueprint(
                tag= masked_query,
                steps = blueprint["steps"],
            )
            self.add_blueprint(agent_blueprint)
            matched_blueprint = agent_blueprint

            print(self.blueprint_db)
            
        return {
            "masked_query": masked_query,
            "blueprint" : matched_blueprint,
            "variables": variables,
            "score": float(max_score)
        }, scores[best_idx]


