import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os
from dotenv import load_dotenv

load_dotenv()

proj_id = os.getenv("API_KEY")


vertexai.init(project=proj_id, location="us-central1") 

def run_vertex_teacher(masked_query):
    model = GenerativeModel("gemini-2.5-pro")
    
    prompt = f"""
    You are an expert AI Systems Architect. 
    Your job is to read a masked user query and design a step-by-step Execution Blueprint
    that a separate, simpler "Worker" AI will follow to solve the problem. 
    

    AVAILABLE TOOLS FOR THE WORKER:
    1. `fetch_document(company, year, target_metric)`: Retrieves the financial PDF. 'target_metric' MUST be the specific financial variable or line item needed (e.g., "Operating Income", "Capital Expenditure").
    2. `calculate_math(expression)`: Computes math safely.
    3. `submit_answer(final_value)`: Returns the final answer to the user and ends the loop.

    STRICT RULES:
    - The output must contain only useful words with no filler text introducing the steps.
    - The steps must be completely GENERALIZED. Do not use specific company names or numbers. 
    - Refer only to the variables present in the masked query (e.g., [ORG], [year], [metric]).
    - Each step must describe the logic and explicitly state which tool the Worker should use.
    - Output MUST be a valid JSON array of strings, where each string is one step.

--- FEW-SHOT EXAMPLE ---
    USER MASKED QUERY: "Calculate the [metric] for [ORG] in [year]."
    YOUR JSON OUTPUT:
    {{
        "steps": [
            "Step 1: Invoke the `fetch_document` tool using [ORG], [year], and [metric] to get the financial context.",
            "Step 2: Scan the retrieved document to find the numerical values that make up the [metric]. If a calculation is required, invoke the `calculate_math` tool with the formula.",
            "Step 3: Invoke the `submit_answer` tool with the final extracted or calculated value."
        ]
    }}
    ------------------------

    ACTUAL MASKED QUERY TO PROCESS: "{masked_query}"
    YOUR JSON OUTPUT:
    """
    
    # Force the model to return strict JSON
    config = GenerationConfig(response_mime_type="application/json")
    
    response = model.generate_content(prompt, generation_config=config)
    
    return response.text

if __name__ == "__main__":
    print(run_vertex_teacher("what is the [year] [financial metric] (in [unit of financial quantity]) for [company]? give a response to the question by relying on the details shown in the [financial data document]."))