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

SYSTEM_PROMPT = """You are a JSON-only financial execution agent. Follow the BLUEPRINT exacty.

RULES:
1. EXHAUSTIVE: If VARIABLES contains a list, process every single item. If data is missing, explicitly state "No data found".
2. RAW NUMBERS: `calculate_math` requires ONLY numbers (e.g., '10/2'). No text or $ symbols.
3. FINAL ANSWER: `submit_answer` must include the actual calculated numbers and entity names, not placeholders.

SCHEMA (You must output ONLY this exact JSON structure):
{
    "thought": "Briefly explain the step",
    "tool": "tool_name",
    "kwargs": {"arg_name": "arg_value"}
}

TOOLS:
1. fetch_document(company, years, target_metrics)
2. calculate_math(expression)
3. submit_answer(final_value)

BLUEPRINT:\n
"""


def execute_blueprint(blueprint_steps, variables, current_row_index=0, max_loops=5):
    """
    Takes the cached steps, injects the variables, and loops the LLM through the tools.

    Takes in a row index corresponding to the finbench sample we are trying.
    """

    # Fed to the agent, constant across all ReAct loop
    # System, blueprint steps in order
    # If variables are in list form, try adjusting system prompt for in order use of the list, otherwise resort to variable substitution.
    static_prompt = SYSTEM_PROMPT + "\n".join([s for s in blueprint_steps]) + "\nVARIABLES:\n" + str(variables.items())
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
