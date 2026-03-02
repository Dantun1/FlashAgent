from utils.finbench_utils import get_questions
from plan_cache_engine import PlanCacheEngine, AgentBlueprint
from agentaction.actions import execute_blueprint

if __name__ == "__main__":
    engine = PlanCacheEngine()
    questions = get_questions(150)
    for idx in range(10):
        print(questions[idx])
        bp = engine.retrieve_plan(questions[idx])
        print(bp["blueprint"].steps)
        print(bp["variables"])
        execute_blueprint(bp["blueprint"].steps, bp["variables"], current_row_index=idx, max_loops=20)

    # acquiq = questions[52]
    # masked_acquiq = "[EXTRACTION] among operations, investing, and financing activities, which brought in the most (or lost the least) [financial metric] for [company] in [year]?"
    # vars = engine._extract_and_mask(acquiq)[1]
    
    # print(masked_acquiq)
    # print(vars)
    # steps = [
    #     "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[year], and target_metrics='[financial metric]'.", 
    #     'Step 2 [SUBMIT]: Read the TOOL OUTPUT of Step 1 to locate the three separate numerical values for [financial metric] from operating, investing, and financing activities. Compare these three numbers to identify which activity has the highest value (i.e., the most positive or least negative number). Invoke `submit_answer` with a full sentence stating [company] brought in the most or lost the least[financial metric] through the identified activity, including the name of the activity and its corresponding value with CORRECT UNITS.'  
    # ]
    
    # execute_blueprint(steps, vars, 52, 20)


    

    