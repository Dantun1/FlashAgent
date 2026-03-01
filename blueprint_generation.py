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
    Your job is to read a masked user query and output a strict, deterministic sequence of steps.

    AVAILABLE TOOLS FOR THE WORKER:
    1. `fetch_document(company, years: list, target_metrics)`: Retrieves financial text relevant to a company for a given set of year for the given set of metrics.
    2. `calculate_math(expression)`: Computes math (+, -, *, /) safely. ONLY accepts raw numbers.
    3. `submit_answer(final_value)`: Submits the final qualitative or quantitative answer and terminates the loop.

    STRICT BLUEPRINT RULES:
    1. **Format & Allowed Actions:** Minimise conversational filler and verbosity. Start every step with the required ACTION tag: `[FETCH]`, `[CALCULATE]`, or `[SUBMIT]`. Do NOT create steps for "extracting" or "analyzing" without attaching them directly to a `[CALCULATE]` or `[SUBMIT]` action.
    2. **The Target Metric Rule:** If the variable `[financial metric]` is in the query, you MUST pass the literal string `"[financial metric]"` as the `target_metric` argument in the fetch tool. Do not invent section names/hallucinate. 
    3 **The Unit Conversion rule** If the query specifies a unit, include a [CALCULATE] statement that requires confirmation of units, and if not, correct conversion applied.
    3. **The Implicit Metric Rule:** If the query asks a conceptual question (e.g., "capital intensive") and lacks a `[financial metric]` variable, you MUST explicitly hardcode the standard financial line items required (e.g., 'Property, Plant, and Equipment' and 'Total Assets').
    4. **The Synthesis Rule:** If the query asks a Yes/No, "Why", or "What drove" question, the final `[SUBMIT]` step MUST explicitly instruct the worker to form a full qualitative sentence using the calculated numbers as proof, use the specification of the prompt to ensure the model covers all details required in the prompt.

    --- FEW-SHOT EXAMPLE 1 (Direct Extraction) ---
    MASKED QUERY: "what is the [year] [financial metric] for [company]?"
    JSON OUTPUT:
    {{
        "steps": [
            "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], year=[year], and target_metric=[financial metric].",
            "Step 2 [SUBMIT]: Read the text output from Step 1 to locate the exact numerical value for [financial metric]. Invoke `submit_answer` with this raw number."
        ]
    }}

    ------------------------

        --- FEW-SHOT EXAMPLE 2 (Multi-year extraction) ---
    MASKED QUERY: "[EXTRACTION] what are major acquisitions that [company] has done in [year], [year] and [year]?"
    JSON OUTPUT:
    {{
        "steps": [
            "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], years=[year], and target_metrics=['Acquisitions'].", 
            'Step 2 [SUBMIT]: Read the text from the TOOL OUTPUTS of the previous step. For each year listed in VARIABLES, identify and list the major acquisitions mentioned. Invoke `submit_answer` with a detailed description of the acqusitions, if any.'
        ]
    }}

    ------------------------
    

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