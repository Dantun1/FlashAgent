from plan_cache_engine import PlanCacheEngine, AgentBlueprint
from finbench_utils import get_questions

engine = PlanCacheEngine()

financebench_analysis_plan = AgentBlueprint(
    description="How much money did apple make in 2018",  
    steps=["some trial steps"]
)

engine.add_blueprint(financebench_analysis_plan)



if __name__ == "__main__":
    questions = get_questions()
    q = questions[0]
    print(q)
    engine.retrieve_plan(q)
