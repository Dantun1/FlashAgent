from utils.finbench_utils import get_questions
from plan_cache_engine import PlanCacheEngine, AgentBlueprint
from agentaction.actions import execute_blueprint

if __name__ == "__main__":
    engine = PlanCacheEngine()
    questions = get_questions(150)
    # for idx in range(10):
        # print(questions[idx])
        # bp = engine.retrieve_plan(questions[idx])
        # print(bp["blueprint"].steps)
        # print(bp["variables"])
        # execute_blueprint(bp["blueprint"].steps, bp["variables"], current_row_index=idx, max_loops=20)

    acquiq = questions[51]
    masked_acquiq = "[EXTRACTION] what are major acquisitions that [company] has done in [year], [year] and [year]?"
    vars = engine._extract_and_mask(acquiq)[1]
    
    print(masked_acquiq)
    print(vars)
    steps = [
            "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[year], and target_metrics=['Acquisitions'].", 
            'Step 2 [SUBMIT]: Read the text from the TOOL OUTPUTS of the previous step. For each year listed in VARIABLES, identify and list the major acquisitions mentioned. Invoke `submit_answer` with a detailed description of the acqusitions, if any.'
        ]
    
    execute_blueprint(steps, vars, 51, 10)
    

    

    