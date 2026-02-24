from plan_cache_engine import PlanCacheEngine, AgentBlueprint, BlueprintStep
from finbench_utils import get_questions

engine = PlanCacheEngine()

financebench_analysis_plan = AgentBlueprint(
    id="financebench_metric_extraction",
    description="analyze the [metric] of [company]",  
    steps=[
        BlueprintStep(
            step_number=1,
            required_variables=["company", "year"]
        ),
        BlueprintStep(
            step_number=2,
            required_variables=["metric"]
        )
    ]
)

engine.add_blueprint(financebench_analysis_plan)



if __name__ == "__main__":
    questions = get_questions()
    print(questions[0])
    masked_q, _ = engine._extract_and_mask(questions[0])

    print(masked_q)
