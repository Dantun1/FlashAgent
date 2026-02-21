from plan_cache_engine import PlanCacheEngine, AgentBlueprint, BlueprintStep
from finbench_utils import get_questions

engine = PlanCacheEngine()

financebench_analysis_plan = AgentBlueprint(
    id="financebench_metric_extraction",
    description="analyze the [metric] of [company]",  
    steps=[
        BlueprintStep(
            instruction="Invoke the document retrieval tool. Pass the target organization and fiscal year to load the correct financial statement into context.",
            required_variables=["company", "year"]
        ),
        BlueprintStep(
            instruction="Search the retrieved context document for the target financial metric. Return the exact numerical value.",
            required_variables=["metric"]
        )
    ]
)

engine.add_blueprint(financebench_analysis_plan)



if __name__ == "__main__":
    questions = get_questions()

    sample_qs = questions[:3]
    for q in sample_qs:
        # Testing the masking
        masked_q, vars = engine._extract_and_mask(q)
        print(vars,sep="\n", end = "\n")
        data = engine.retrieve_plan(q)
        print(data["matched_id"])