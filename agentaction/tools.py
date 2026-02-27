import re

from utils.finbench_utils import load_finbench

financebench_data = load_finbench()


def fetch_document(company: str, year: str, target_metric: str, current_row_index: int = 0) -> str:

    print(f"Fetching document for {company} | {year} | {target_metric}")
    
    try:
        current_row = financebench_data.iloc[current_row_index]


        gt_company = str(current_row.get("company", "")).lower()
        gt_year = str(current_row.get("financial_year", "")).lower()
        
        # Check for expected company
        if company.lower() not in gt_company and gt_company not in company.lower():
            return f"ERROR: Database lookup failed. No filings found for company '{company}'."
            
        
        # Fetch evidence for the query

        evidence_list = current_row["evidence"] 
        evidence_text = evidence_list[0]["evidence_text_full_page"]
        cleaned_text = re.sub(r'\n\s*\n', '\n', evidence_text)
        cleaned_text = re.sub(r' \s+', ' ', cleaned_text)
        
        return f"DOCUMENT CONTEXT (Page {evidence_list[0]['evidence_page_num']}):\n{cleaned_text}"
        
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
