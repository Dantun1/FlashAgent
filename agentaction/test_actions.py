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

    acquiq = questions[2]
    masked_acquiq = "[EXTRACTION] is [company] a capital-intensive business based on [year] [financial data document]?"
    vars = engine._extract_and_mask(acquiq)[1]
    
    print(masked_acquiq)
    print(vars)
    steps = [
        "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[year], and target_metrics=['Capital Expenditures', 'Revenue', 'Property, Plant, and Equipment', 'Total Assets', 'Net Income']", 
        "Step 2 [CALCULATE]: Read the TOOL OUTPUT of Step 1 to find the numerical values for 'Purchases of property, plant and equipment' (this is CAPEX) and 'Net sales' (this is Revenue).  Invoke `calculate_math` to find the CAPEX/Revenue ratio using the previously extracted numbers.",
        "Step 3 [CALCULATE]: Read the TOOL OUTPUT of Step 1 to find the numerical values for 'Property, plant and equipment net' (this is Fixed assets) and 'Total assets' (this is Total Assets).  Invoke `calculate_math` to find the Fixed assets/Total assets ratio using the previously extracted numbers.",
        "Step 4 [CALCULATE]: Read the TOOL OUTPUT of Step 1 to find the numerical values for 'Net income attributable to 3M' and 'Total assets'. Invoke `calculate_math` to find the Net Income / Total assets (this is the Return on Assets (ROA)) using the previously extracted numbers..",
        "Step 5 [SUBMIT]: Analyze the TOOL OUTPUTS of Step 1, Step 2, and Step 3. Then, invoke `submit_answer` with 1 sentence stating if the company is capital intensive based on these ratios, followed by a supporting sentence referencing the list of the exact calculated metrics from previous steps"
    ]
    
    execute_blueprint(steps, vars, 2, 20)


    

    