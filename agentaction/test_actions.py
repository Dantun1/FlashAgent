from utils.finbench_utils import get_questions
from plan_cache_engine import PlanCacheEngine, AgentBlueprint
from agentaction.actions import execute_blueprint

if __name__ == "__main__":
    engine = PlanCacheEngine()
    questions = get_questions(10)
    for idx in range(10):
        print(questions[idx])
        bp = engine.retrieve_plan(questions[idx])
        print(bp["blueprint"].steps)
        print(bp["variables"])
        execute_blueprint(bp["blueprint"].steps, bp["variables"], current_row_index=idx, max_loops=20)


    