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
    1. `fetch_document(company, year, target_metric)`: Retrieves financial text.
    2. `calculate_math(expression)`: Computes math (+, -, *, /) safely.
    3. `submit_answer(final_value)`: Submits the final answer and terminates the loop.

    STRICT BLUEPRINT RULES:
    1. **Format:** Do not use conversational filler. Start every step with the required ACTION in brackets (e.g., `Step 1 [FETCH]: ...`).
    2. **Variables:** You MUST use the exact bracketed variables from the query (e.g., `[company]`, `[year]`).
    3. **Missing Metrics (The Analytical Rule):** If the query asks a conceptual question (e.g., "Is it capital intensive?", "What drove margins?"), the query will not contain a `[metric]` variable. You MUST explicitly hardcode the standard financial line items required to solve it (e.g., 'Property, Plant, and Equipment' and 'Total Assets') as the `target_metric` strings in the steps.
    4. **The Synthesis Rule:** If the query asks a Yes/No or "Why" question, the final step MUST explicitly instruct the worker on how to interpret the math to form a sentence. Do NOT just submit a raw number.

    --- FEW-SHOT EXAMPLE 1 (Direct Extraction) ---
    MASKED QUERY: "what is the [year] [financial metric] for [company]?"
    JSON OUTPUT:
    {{
        "steps": [
            "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], year=[year], and target_metric=[financial metric].",
            "Step 2 [EXTRACT]: Read the text output from Step 1 to locate the exact numerical value for [financial metric].",
            "Step 3 [SUBMIT]: Invoke `submit_answer` with the extracted numerical value."
        ]
    }}

--- FEW-SHOT EXAMPLE 2 (Qualitative/Derived Question) ---
    MASKED QUERY: "is [company] a capital-intensive business based on [year] data?"
    JSON OUTPUT:
    {{
        "steps": [
            "Step 1 [FETCH]: Invoke `fetch_document` with company=[company], year=[year], and target_metric='Property, Plant, and Equipment'.",
            "Step 2 [EXTRACT]: Read the text output from Step 1 and identify the exact numerical value for Property, Plant, and Equipment.",
            "Step 3 [FETCH]: Invoke `fetch_document` with company=[company], year=[year], and target_metric='Total Assets'.",
            "Step 4 [EXTRACT]: Read the text output from Step 3 and identify the exact numerical value for Total Assets.",
            "Step 5 [CALCULATE]: Invoke `calculate_math` to divide the raw number extracted in Step 2 by the raw number extracted in Step 4 (e.g., '100 / 500').",
            "Step 6 [SYNTHESIZE]: Analyze the calculated ratio. If the ratio is high (e.g., > 0.25), it is capital intensive. Invoke `submit_answer` with a full sentence stating YES or NO, including the calculated ratio as proof."
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