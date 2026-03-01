import json
import time
from openai import OpenAI

from agentaction.tools import fetch_document, calculate_math, submit_answer

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local-mac")
MODEL_NAME = "llama3.1"

AVAILABLE_TOOLS = {
    "fetch_document": fetch_document,
    "calculate_math": calculate_math,
    "submit_answer": submit_answer
}

SYSTEM_PROMPT = f"""You are a financial execution agent. 
You have been provided TOOLS and a STRICT execution blueprint. You must follow the steps exactly and 
use the provided variable VALUES to generate TOOL CALLS by filling in the placeholders in the blueprint. Your output must obey
this schema:

SCHEMA (You must output ONLY this exact JSON structure):
{{
    "thought": "Briefly explain the step",
    "tool": "tool_name",
    "kwargs": {{"arg_name": "arg_value"}}
}}


STRICT RULES:


1. When a document is fetched, carefully scan the text for SPECIFIED INFORMATION. 
2. Financial terms may be synonyms (e.g., "Capital Expenditure" is often listed as "Purchases of property, plant and equipment").
3. Do not include markdown blocks or any other text. Just the JSON.
4. SYNTHESIS RULE: When submitting your final answer, you MUST substitute the actual numerical values you calculated into your sentence. NEVER output placeholder text like "[insert value]". Read your previous tool outputs and use the actual numbers.

AVAILABLE TOOLS:
1. fetch_document(company, years, target_metrics) : Fetches specified document portion given a company, list of years and list of target metrics
2. calculate_math(expression) : ONLY use for arithmetic (+, -, *, /) with RAW NUMBERS (e.g., "1500 / 10"). STRICTLY PROHIBITED: Do not put text, variables, conditions, or function calls (like fetch_document) inside the expression.
3. submit_answer(final_value) : : Submits the final answer. 'final_value' MUST be a single, flat text string containing either the qualitative or numeric response IN DETAIL. 
BLUEPRINT STEPS:\n
"""


def execute_blueprint(blueprint_steps, variables, current_row_index=0, max_loops=5):
    """
    Takes the cached steps, injects the variables, and loops the LLM through the tools.

    Takes in a row index corresponding to the finbench sample we are trying.
    """

    # Fed to the agent, constant across all ReAct loop
    # System, blueprint steps in order
    # If variables are in list form, try adjusting system prompt for in order use of the list, otherwise resort to variable substitution.
    vars_string = json.dumps(dict(variables), indent=2)

    static_prompt = SYSTEM_PROMPT + "\n".join([s for s in blueprint_steps]) + "\nVARIABLES:\n" + vars_string + "ACTION FOR STEP 1:"
    print(static_prompt)
    messages = [{"role": "system", "content": static_prompt}]

    # Iteratively query for actions
    for loop_idx in range(max_loops):
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        llm_output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": llm_output})
        
        try:
            # Parse the LLM's requested action
            action = json.loads(llm_output)
            print(action.get("thought"))
            tool_name = action.get("tool")
            kwargs = action.get("kwargs", {})
            
            print(f"Requested: {tool_name}({kwargs})")
            
            if tool_name not in AVAILABLE_TOOLS:
                tool_result = f"ERROR: Tool {tool_name} not found."
            else:
                if tool_name == "fetch_document":
                    kwargs["current_row_index"] = current_row_index
            
                try:
                    tool_result = AVAILABLE_TOOLS[tool_name](**kwargs)
                except TypeError as e:
                    tool_result = f"TOOL ARGUMENT ERROR: Wrong arguments passed to {tool_name}. Details: {str(e)}"

            print(tool_result)
            if tool_name == "submit_answer":
                print(f"\nFINAL ANSWER: {tool_result}")
                return tool_result
            # Otherwise, feed the tool output back to the LLM
            messages.append({"role": "user", "content": f"TOOL OUTPUT:\n{tool_result}\n\nProceed to the next step."})
            
        except json.JSONDecodeError:
            print(f"JSON Parse Error. Raw Output: {llm_output}")
            messages.append({"role": "user", "content": "ERROR: You must output ONLY valid JSON. Try again."})
            
    print("\nReached max loops without submitting an answer.")

    return None
