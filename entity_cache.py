from plan_cache_engine import PlanCacheEngine, AgentBlueprint
from finbench_utils import get_questions

engine = PlanCacheEngine()

financebench_analysis_plan = AgentBlueprint(
    tag="How much money did apple make in FY2018, look at the cash flow statement",  
    steps=["some trial steps"]
)

engine.add_blueprint(financebench_analysis_plan)

if __name__ == "__main__":
    questions = get_questions()
    q = questions[0]
    engine.retrieve_plan(q)
