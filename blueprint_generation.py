import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import os
from dotenv import load_dotenv

load_dotenv()

proj_id = os.getenv("API_KEY")


vertexai.init(project=proj_id, location="us-central1") 

def run_vertex_teacher(masked_query):
    print("Cache Miss! Routing to Vertex AI...")
    model = GenerativeModel("gemini-2.5-pro")
    
    prompt = f"""
    You are an expert AI Systems Architect. 
    Your job is to read a masked user query and design a step-by-step Execution Blueprint 
    that a separate, simpler "Worker" AI will follow to solve the problem.

    AVAILABLE TOOLS FOR THE WORKER:
    1. `fetch_document(company, year, metric_context)`: Retrieves the financial PDF.
    2. `calculate_math(expression)`: Computes math safely.
    3. `submit_answer(final_value)`: Returns the final answer to the user and ends the loop.

    STRICT RULES:
    - The blueprint must be completely GENERALIZED. Do not use specific company names or numbers. 
    - Refer only to the variables present in the masked query (e.g., [ORG], [year], [metric]).
    - Each step must describe the logic and explicitly state which tool the Worker should use.
    - Output MUST be a valid JSON array of strings, where each string is one step.

    --- FEW-SHOT EXAMPLE ---
    USER MASKED QUERY: "Calculate the operating margin for [ORG] in [year]."
    YOUR JSON OUTPUT:
    [
        "Step 1: Invoke the `fetch_document` tool using the target [ORG] and [year] to get the financial context.",
        "Step 2: Scan the retrieved document to find the Operating Income and Total Revenue. Invoke the `calculate_math` tool to divide Operating Income by Total Revenue.",
        "Step 3: Format the calculated result as a percentage and invoke the `submit_answer` tool with the final value."
    ]
    ------------------------

    ACTUAL MASKED QUERY TO PROCESS: "{masked_query}"
    YOUR JSON OUTPUT:
    """
    
    # Force the model to return strict JSON
    config = GenerationConfig(response_mime_type="application/json")
    
    response = model.generate_content(prompt, generation_config=config)
    
    return response.text


print(run_vertex_teacher("what is the [year] [financial metric] (in [unit of financial quantity]) for [company]? give a response to the question by relying on the details shown in the [financial data document]."))