import json
import time
import csv
import re
from openai import OpenAI
from agentaction.tools import fetch_document, calculate_math, submit_answer

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"

AVAILABLE_TOOLS = {
    "fetch_document": fetch_document,
    "calculate_math": calculate_math,
    "submit_answer": submit_answer
}

SYSTEM_PROMPT = """You are a highly logical financial execution agent. 
You have been provided TOOLS and a STRICT execution blueprint. You must follow the steps sequentially.
IMPORTANT: The blueprint is generic and consists of VARIABLES wrapped with [], e.g. [company]; You must ALWAYS substitute these values with real values provided to you 
by your minion.

Generate your action by outputting ONLY valid JSON matching this exact schema:

{
    "thought": "Explain your current objective, and the use the data you have in relation to this goal to justify your answer",
    "tool": "tool_name",
    "kwargs": {"arg_name": "arg_value"}
}

STRICT RULES:
1. Read the TOOL OUTPUTS carefully to extract required numbers. Financial terms may be synonyms (e.g., "Capital Expenditure" = "Purchases of property, plant and equipment").
2. NEVER wrap your output in markdown formatting (like ```json). Output just the raw JSON object.
3. SYNTHESIS RULE: When using `submit_answer`, you MUST substitute the actual numerical values you calculated into your sentence. NEVER output placeholder text.
4. MISSING DATA RULE: If the Minion returns 'DATA NOT FOUND' for a required metric, DO NOT attempt to calculate math using fake or duplicate numbers. You must immediately invoke `submit_answer` stating that the calculation cannot be completed due to missing data.

AVAILABLE TOOLS:
1. fetch_document(company, years, target_metrics) : Fetches document portion.
2. calculate_math(expression) : ONLY use for arithmetic (+, -, *, /) with RAW NUMBERS (e.g., "1500 / 10"). NO TEXT OR VARIABLES.
3. submit_answer(final_value) : Submits the final answer string in detail.

BLUEPRINT STEPS:\n"""



def call_minion_extractor(document_text, company, years, target_metrics, blueprint):
    """
    isolated/stateless document data extractor call rto pass data to the actual actor model
    """
    minion_prompt = f"""You are a precise financial extraction minion.
Your ONLY job is to read the provided financial document and extract ALL the exact numerical values needed to successfully execute the following BLUEPRINT.
We have provided you a list of specifically requested metrics, but this is NOT exhaustive. Retrieve anything that is listed or adjacent to the information in the blueprint.

Blueprint: {blueprint}
Company: {company}
Years: {years}
Explicitly specified metrics to extract: {target_metrics}

RULES:
1. Return ONLY the extracted numbers with their associated labels and units (e.g., "Apple 2021 Revenue: $365 million").
2. IMPORTANT: Financial metrics use different names. If you don't see the exact term, look for standard accounting synonyms (e.g., "Cost of Revenue" might be "Cost of Sales", "Dividends" might be "Cash dividends declared"). 
3. Only output 'DATA NOT FOUND' if you have exhaustively checked for all possible synonyms.
4. Do not include any conversational filler.

--- DOCUMENT CONTEXT ---
{document_text}
"""
    print(document_text)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": minion_prompt}],
        temperature=0.0,
        max_tokens=200 
    )
    
    return response.choices[0].message.content

kv_tracking_cols = [
                    "idx", 
                    "tool_call_num",
                    "total_tokens",
                    "cached_tokens",
                    "prefill_tokens"
    ]


def execute_blueprint(blueprint_steps, variables, current_row_index=0, max_loops=5):
    """
    Takes the cached steps, adapts them with variables, and loops the LLM through the tools.
    """
    ##
    # 1. BLUEPRINT ADAPTATION (Stateful Sequential Interpolation)
    # adapted_steps = []
    #
    # # Track how many times we've used each label across the ENTIRE blueprint
    # label_counters = {label: 0 for label in variables.keys()}
    #
    # # Extract the base year as an integer if it exists (e.g., "fy2022" -> 2022)
    # base_year = None
    # if "year" in variables and variables["year"]:
    #     for y in variables["year"]:
    #         match = re.search(r'\d{4}', str(y))
    #         if match:
    #             base_year = int(match.group())
    #             break
    #
    # for step in blueprint_steps:
    #     adapted_step = step
    #
    #     # A. Resolve relative years like [year-1], [year-2] dynamically
    #     if base_year is not None:
    #         def replace_relative_year(m):
        #         offset = int(m.group(1))
        #         return str(base_year - offset)
        #
        #     adapted_step = re.sub(r'\[year\s*-\s*(\d+)\]', replace_relative_year, adapted_step, flags=re.IGNORECASE)
        #
        # # B. Inject standard variables sequentially
        # for label, val_list in variables.items():
        #     if not val_list:
        #         continue
        #
            # # 1. Handle explicit numbered placeholders (e.g., [financial metric 1])
            # for i, val in enumerate(val_list):
            #     adapted_step = adapted_step.replace(f"[{label} {i+1}]", str(val))
            #
            # # 2. Sequential replacement for generic placeholders (e.g., [financial metric])
            # tag = f"[{label}]"
            # while tag in adapted_step:
            #     # Use modulo to cycle through the list safely if there are more tags than values
            #     idx = label_counters[label] % len(val_list)
            #     current_val = str(val_list[idx])
                
                # Replace ONLY the FIRST occurrence in the string
                # adapted_step = adapted_step.replace(tag, current_val, 1)
                
                # Increment the counter so the next occurrence gets the next value!
                # label_counters[label] += 1
                
        # adapted_steps.append(adapted_step)
    ##
    static_prompt = SYSTEM_PROMPT + "\n".join(blueprint_steps) +"\nVARIABLES:\n" +json.dumps(dict(variables), indent = 2)+"\n\nACTION FOR STEP 1:"
    messages = [{"role": "system", "content": static_prompt}]

    for loop_idx in range(max_loops):
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        llm_output = response.choices[0].message.content
        usage = response.usage
        total_output = usage.prompt_tokens

      
        cached_tokens = usage.prompt_tokens_details.cached_tokens
        actual_prefill = total_output - cached_tokens

        with open("kv_tracking.csv", "a") as csvfile:
            writer = csv.DictWriter(csvfile,kv_tracking_cols)
            writer.writerow({"idx":current_row_index,"tool_call_num":loop_idx, "total_tokens":total_output, "cached_tokens":cached_tokens, "prefill_tokens": actual_prefill}) 
        


        messages.append({"role": "assistant", "content": llm_output})
        
        try:
            action = json.loads(llm_output)
            tool_name = action.get("tool")
            kwargs = action.get("kwargs", {})
            
            if tool_name not in AVAILABLE_TOOLS:
                tool_result = f"ERROR: Tool {tool_name} not found."
            else:
                if tool_name == "fetch_document":
                    kwargs["current_row_index"] = current_row_index
                    
                    raw_document = AVAILABLE_TOOLS[tool_name](**kwargs)
                    
                    print(f"\n[MINION] Reading document to extract info for blueprint, especially {kwargs.get('target_metrics')}...")
                    tool_result = call_minion_extractor(
                        document_text=raw_document,
                        company=kwargs.get("company"),
                        years=kwargs.get("years"),
                        target_metrics=kwargs.get("target_metrics"),
                        blueprint = "\n".join(blueprint_steps),
                    )
                    print(f"[MINION OUTPUT]: {tool_result}")
                    
                else:
                    try:
                        tool_result = AVAILABLE_TOOLS[tool_name](**kwargs)
                    except TypeError as e:
                        tool_result = f"TOOL ARGUMENT ERROR: Wrong arguments passed. Details: {str(e)}"

            if tool_name == "submit_answer":
                print(f"\nFINAL ANSWER: {tool_result}")
                return tool_result
            
            messages.append({
                "role": "user", 
                "content": f"TOOL OUTPUT:\n{tool_result}\n\nProceed to the exact next step in the blueprint."
            })
            
        except json.JSONDecodeError:
            print(f"JSON Parse Error. Raw Output: {llm_output}")
            messages.append({
                "role": "user", 
                "content": "ERROR: You must output ONLY raw, valid JSON. Do not include markdown formatting or explanations outside the JSON."
            })
            
    print("\nReached max loops without submitting an answer.")
    return None
