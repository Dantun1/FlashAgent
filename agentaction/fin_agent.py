import json
import csv
import re
from openai import OpenAI

from agentaction.tools import FinanceToolkit
from plan_cache_engine import PlanCacheEngine, AgentBlueprint


class FinancialAgent:
    """
    Financial Agent that responds to FinanceBench queries using semantic plan caching.
    """
    MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
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

    MINION_PROMPT = """You are a precise financial extraction minion.
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
    KV_TRACKING_COLS = ["idx", 
                        "tool_call_num",
                        "total_tokens",
                        "cached_tokens",
                        "prefill_tokens"
    ]

    def __init__(self, cache_prefill_info: dict[str, str] = None, cache_enabled: bool = True, model: str = None) -> None:
        # if cache is disabled, every query is treated as cache miss
        self._cache = PlanCacheEngine(cache_enabled)
        # prefill if provided
        if cache_prefill_info is not None:
            masked_queries = list(cache_prefill_info.keys())
            blueprints = list(cache_prefill_info.values())
            
            self._cache.prefill_cache(masked_queries,blueprints)
        
        # Connection to inference engine
        self._inference_router = OpenAI(base_url="http://localhost:30000/v1", api_key = "EMPTY")
        if model is not None:
            self.MODEL_NAME = model

        self._toolkit = FinanceToolkit()

    def run(self, query: str, idx: int) -> str:
        """
        Executes a query using the configured cache settings.

        Note: we take an index param, this is because we are mocking tool calls via the FinBench dataset.
        In production, we would have a RAG system where the required data is fetched quickly/consistently

        Arguments:
            query: the finbench question to ask
            idx: the index of the query in the dataset
        """
        # Fetch the plan info
        plan_info = self._cache.retrieve_plan(query)
        # Run inference
        result = self._execute_blueprint(plan_info["blueprint"].steps, plan_info["variables"], current_row_index = idx, max_loops = 20)
        
        return result
    
    def _execute_blueprint(self, blueprint_steps, variables, current_row_index, max_loops=10):
        """
        Runs execution loop, generating actions until submit_answer() is invoked by the agent
        """
        
        static_prompt = self.SYSTEM_PROMPT + "\n".join(blueprint_steps) +"\nVARIABLES:\n" +json.dumps(dict(variables), indent = 2)+"\n\nACTION FOR STEP 1:"
        messages = [{"role": "system", "content": static_prompt}]

        for loop_idx in range(max_loops):
        
            response = self._inference_router.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
        
            llm_output = response.choices[0].message.content

            # Extract usage info for tracking
            usage = response.usage

            total_tokens = usage.prompt_tokens
            cached_tokens = usage.prompt_tokens_details.cached_tokens
            actual_prefill = total_tokens - cached_tokens

            with open("kv_tracking.csv", "a") as csvfile:
                writer = csv.DictWriter(csvfile,self.KV_TRACKING_COLS)
                writer.writerow({"idx":current_row_index,"tool_call_num":loop_idx, "total_tokens":total_tokens, "cached_tokens":cached_tokens, "prefill_tokens": actual_prefill}) 
        

            # Record attempted action, attempt to execute
            messages.append({"role": "assistant", "content": llm_output})
            try:
                action = json.loads(llm_output)
                tool_name = action.get("tool")
                kwargs = action.get("kwargs", {})
            
                if tool_name not in self._toolkit.AVAILABLE_TOOLS:
                    tool_result = f"ERROR: Tool {tool_name} not found."
                else:
                    if tool_name == "fetch_document":
                        kwargs["current_row_index"] = current_row_index
                    
                        raw_document = self._toolkit.fetch_document(**kwargs)
                        
                        # Call stateless minion to extract relevant data
                        tool_result = self._call_minion_extractor(
                            document_text=raw_document,
                            company=kwargs.get("company"),
                            years=kwargs.get("years"),
                            target_metrics=kwargs.get("target_metrics"),
                            blueprint = "\n".join(blueprint_steps),
                        )
                    
                    else:
                        try:
                            # either calculate or submit
                            tool_result = self._toolkit.AVAILABLE_TOOLS[tool_name](**kwargs)
                        except TypeError as e:
                            tool_result = f"TOOL ARGUMENT ERROR: Wrong arguments passed. Details: {str(e)}"
                    
                    # Break loop if submitted
                    if tool_name == "submit_answer":
                        return tool_result
                    
                    messages.append({
                        "role": "user", 
                        "content": f"TOOL OUTPUT:\n{tool_result}\n\nProceed to the exact next step in the blueprint."
                    })
            
            except json.JSONDecodeError:
                messages.append({
                    "role": "user", 
                    "content": "ERROR: You must output ONLY raw, valid JSON. Do not include markdown formatting or explanations outside the JSON."
                })
            
        return None

    def _call_minion_extractor(self, document_text, company, years, target_metrics, blueprint):
        """
        isolated/stateless document data extractor call rto pass data to the actual actor model
        """
        formatted_prompt = self.MINION_PROMPT.format(
            blueprint=blueprint,
            company=company,
            years=years,
            target_metrics=target_metrics,
            document_text=document_text
        )
        response = self._inference_router.chat.completions.create(
        model=self.MODEL_NAME,
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.0,
        max_tokens=200 
        )
    
        return response.choices[0].message.content
