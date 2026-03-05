import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os
from dotenv import load_dotenv

load_dotenv()

proj_id = os.getenv("PROJECT_ID")


vertexai.init(project=proj_id, location="us-central1") 

def generate_new_blueprint(masked_query):
    model = GenerativeModel("gemini-2.5-pro")
    
    prompt = f"""
    You are an expert AI Systems Architect designing Execution Blueprints for a rigid, robotic 8B-parameter Worker AI.
    Your job is to read a masked user query and output a strict, deterministic sequence of steps that is unambiguous.

    AVAILABLE TOOLS FOR THE WORKER:
    1. `fetch_document(company, years: list, target_metrics)`: Retrieves financial text relevant to a company for a given set of year for the given set of metrics. Call this function at most ONCE for retrieval of all necessary information at once.
    2. `calculate_math(expression)`: Computes math (+, -, *, /) safely. ONLY accepts raw numbers.
    3. `submit_answer(final_value)`: Submits the final qualitative or quantitative answer and terminates the loop.

    STRICT BLUEPRINT RULES:
    1. **Format & Allowed Actions:** Minimise conversational filler and verbosity. Start every step with the required ACTION tag: `[FETCH]`, `[CALCULATE]`, or `[SUBMIT]`.
    3 **The Unit Conversion rule** If the query specifies a unit, include a [CALCULATE] statement that requires confirmation of units, and if not, correct conversion applied.
    3. **The Implicit Metric Rule:** If the query asks a conceptual question (e.g., "capital intensive"), you must infer the relevant financial statistics/ratios typically used for such analysis and use calculate statements to calculate the required metrics, giving them a name so the model knows how to refer to them.
    4. **The Serialized Math & No Scratchpad Rule:** The worker MUST extract numbers and calculate the result in a SINGLE expression (e.g., '(100 / 50)'). Do NOT allow the worker to use `calculate_math` just to extract a single number. For multi-year calculations (like averages or YoY), force the worker to process ONE YEAR PER STEP before aggregating to prevent table misreading.
    5. **The Synthesis Rule:** If the query asks a Yes/No, "Why", or "What drove" question, the final `[SUBMIT]` step MUST explicitly instruct the worker to form a full qualitative response with 2 sentences using any calculated numbers as proof or directing their logic, use the specification of the prompt to ensure all necessary details are discussed.


    ------------------------

    --- FEW-SHOT EXAMPLE 1 (Multi-year extraction) ---
    MASKED QUERY: "[EXTRACTION] what are major acquisitions that [company] has done in [year], [year] and [year]?"
    JSON OUTPUT:
    {{
        steps = [
            "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[year], and target_metrics=['Acquisitions'].", 
            'Step 2 [SUBMIT]: Read the text from the TOOL OUTPUTS of the previous step. For each year listed in VARIABLES, identify and list the major acquisitions mentioned. Invoke `submit_answer` with a detailed description of the acqusitions, if any.'
        ]
    }}

    ------------------------
    
    --- FEW-SHOT EXAMPLE 2 (Logical reasoning based on implicit numerical evidence)
    MASKED QUERY: "[EXTRACTION] is [company] a capital-intensive business based on [year] [financial data document]?"
    JSON OUTPUT:
    {{
    steps = [
        "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[year], and target_metrics=['Capital Expenditures', 'Revenue', 'Property, Plant, and Equipment', 'Total Assets', 'Net Income']", 
        "Step 2 [CALCULATE]: Read the TOOL OUTPUT of Step 1 to find the numerical values for 'Purchases of property, plant and equipment' (this is CAPEX) and 'Net sales' (this is Revenue).  Invoke `calculate_math` to find the CAPEX/Revenue ratio using the previously extracted numbers.",
        "Step 3 [CALCULATE]: Read the TOOL OUTPUT of Step 1 to find the numerical values for 'Property, plant and equipment net' (this is Fixed assets) and 'Total assets' (this is Total Assets).  Invoke `calculate_math` to find the Fixed assets/Total assets ratio using the previously extracted numbers.",
        "Step 4 [CALCULATE]: Read the TOOL OUTPUT of Step 1 to find the numerical values for 'Net income attributable to [company]' and 'Total assets'. Invoke `calculate_math` to find the Net Income / Total assets (this is the Return on Assets (ROA)) using the previously extracted numbers..",
        "Step 5 [SUBMIT]: Analyze the TOOL OUTPUTS of Step 1, Step 2, and Step 3. Then, invoke `submit_answer` with a SUMMARY LOGICAL CONCLUSION stating EXPLICITLY if the company is capital intensive, then supporting sentence saying this is based on the list of the exact calculated metrics from previous steps"
    ]
    }}

    -------------------------

    --- FEW-SHOT EXAMPLE 3 (Numerical Reasoning)
    MASKED QUERY: "[EXTRACTION] among operations, investing, and financing activities, which brought in the most (or lost the least) [financial metric] for [company] in [year]?"
    JSON OUTPUT:
        {{
    steps = [
        "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[year], and target_metrics='[financial metric]'.", 
        'Step 2 [SUBMIT]: Read the TOOL OUTPUT of Step 1 to locate the three separate numerical values for [financial metric] from operating, investing, and financing activities. Compare these three numbers to identify which activity has the highest value (i.e., the most positive or least negative number). Invoke `submit_answer` with a full sentence stating [company] brought in the most or lost the least[financial metric] through the identified activity, including the name of the activity and its corresponding value with CORRECT UNITS.'  
    ]
    }}

    -------------------------

    --- FEW-SHOT EXAMPLE 4 (Serialized Multi-Year Math) ---
    MASKED QUERY: "[MATH: AVERAGE] what is the [year] - [year] 3 year average of [financial metric 1] as a % of [financial metric 2] for [company]?"
    JSON OUTPUT:
    {{
        "steps": [
            "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[[year], [year-1], [year-2]], and target_metrics=['[financial metric 1]', '[financial metric 2]'].",
            "Step 2 [CALCULATE]: Read the TOOL OUTPUT for the most recent year ONLY. Find the values for '[financial metric 1]' and '[financial metric 2]'. Substitute them into the expression '(Numerator / Denominator)' and invoke `calculate_math`. Name this Ratio 1.",
            "Step 3 [CALCULATE]: Read the TOOL OUTPUT for the middle year ONLY. Substitute the values into the expression '(Numerator / Denominator)' and invoke `calculate_math`. Name this Ratio 2.",
            "Step 4 [CALCULATE]: Read the TOOL OUTPUT for the oldest year ONLY. Substitute the values into the expression '(Numerator / Denominator)' and invoke `calculate_math`. Name this Ratio 3.",
            "Step 5 [CALCULATE]: Construct a math expression to average the three ratios and multiply by 100 to make it a percentage (e.g., '((0.05 + 0.06 + 0.04) / 3) * 100'). Invoke `calculate_math`.",
            "Step 6 [SUBMIT]: Read the final percentage from Step 5. Invoke `submit_answer` stating the final 3-year average percentage."
        ]
    }}

    ACTUAL MASKED QUERY TO PROCESS: "{masked_query}"
    YOUR JSON OUTPUT:
    """
    
    # Force the model to return strict JSON
    config = GenerationConfig(response_mime_type="application/json")
    
    response = model.generate_content(prompt, generation_config=config)

    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count

    return response.text, input_tokens, output_tokens


if __name__ == "__main__":
    print(generate_new_blueprint("what is the [year] [financial metric] (in [unit of financial quantity]) for [company]? give a response to the question by relying on the details shown in the [financial data document]."))