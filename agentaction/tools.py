import re

from utils.finbench_utils import load_finbench




class FinanceToolkit:
    """
    This class encapsulates tools provided to agent.

    It is just a simulation, in production would have some RAG element for precise, fast retrieval
    """
    _finance_data = load_finbench()


    def __init__(self):

        self.AVAILABLE_TOOLS = {
            "fetch_document": self.fetch_document,
            "calculate_math": self.calculate_math,
            "submit_answer": self.submit_answer
        }

    def fetch_document(self, company: str, years: list, target_metrics: list, current_row_index: int = 0) -> str:
        """
        Simulate document fetch from finbench, given a company, years and metrics requested
        """
        try:
            current_row = self._finance_data.iloc[current_row_index]

            row_company = str(current_row.get("company", "")).lower()

            if company.lower() != row_company:
                return f"ERROR: Database lookup failed. No filings found for company '{company}', check again"
            
            evidence_list = current_row["evidence"]

            full_context = ""
            
            for ev in evidence_list:
                page_num = ev.get("evidence_page_num", "Unknown")
                raw_text = ev.get("evidence_text_full_page", "")

                cleaned_text = re.sub(r'\n\s*\n','\n', raw_text)
                cleaned_text = re.sub(r' \s+', ' ', cleaned_text)

                full_context += f"--- DOCUMENT CONTEXT (Page {page_num}) ---\n{cleaned_text}\n\n" 
            return full_context.strip()
        except Exception as e:
            return f"ERROR FETCHING DOCUMENT: {str(e)}"
    
    def calculate_math(self, expression: str) -> str:
        """
        Performs simple math expressions using eval

        Arguments:
            expression: raw expression requested by agent

        Returns:
            str: response to agent
        """
        try:
            clean_expr = str(expression).replace('$', '').replace(',', '')

            allowed_names = {
                "abs": abs, 
                "round": round, 
                "min": min, 
                "max": max
            }
            result = eval(clean_expr, {"__builtins__": {}}, allowed_names)
        
            return f"CALCULATION RESULT: {round(result, 4)}"

        except ZeroDivisionError:
            return "CALCULATION ERROR: Attempted to divide by zero. Check your denominator."

        except Exception as e:
            return f"CALCULATION ERROR: Invalid math expression '{expression}'. Detailed error: {str(e)}"

    def submit_answer(self, final_value):
        return str(final_value)






def fetch_document(company: str, years: list, target_metrics: list, current_row_index: int = 0) -> str:

    print(f"Fetching document for {company} | {years} | {target_metrics}")
    
    try:
        current_row = financebench_data.iloc[current_row_index]

        gt_company = str(current_row.get("company", "")).lower()

        if company.lower() not in gt_company and gt_company not in company.lower():
            return f"ERROR: Database lookup failed. No filings found for company '{company}'."

        evidence_list = current_row["evidence"] 
        
        combined_context = ""
        
        for ev in evidence_list:
            page_num = ev.get("evidence_page_num", "Unknown")
            raw_text = ev.get("evidence_text_full_page", "")

            cleaned_text = re.sub(r'\n\s*\n', '\n', raw_text)
            cleaned_text = re.sub(r' \s+', ' ', cleaned_text)
            
            combined_context += f"--- DOCUMENT CONTEXT (Page {page_num}) ---\n{cleaned_text}\n\n"
            
        return combined_context.strip()
        
    except Exception as e:
        return f"ERROR FETCHING DOCUMENT: {str(e)}"

def calculate_math(expression: str) -> str:

    print(f"Calculating math -> {expression}")
    
    try:
        clean_expr = str(expression).replace('$', '').replace(',', '')

        allowed_names = {
            "abs": abs, 
            "round": round, 
            "min": min, 
            "max": max
        }
        result = eval(clean_expr, {"__builtins__": {}}, allowed_names)
        
        return f"CALCULATION RESULT: {round(result, 4)}"
        
    except ZeroDivisionError:
        return "CALCULATION ERROR: Attempted to divide by zero. Check your denominator."
    except Exception as e:
        return f"CALCULATION ERROR: Invalid math expression '{expression}'. Detailed error: {str(e)}"

def submit_answer(final_value):
    return str(final_value)
