import csv
import ast
import time
from utils.finbench_utils import get_questions
from agentaction.fin_agent import FinancialAgent

def load_cache_from_csv(csv_path: str) -> dict:
    """Reads the CSV and builds the prefill dictionary."""
    prefill_data = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows or cache misses
            if not row.get('matched_blueprint_steps'):
                continue

            masked_query = row['key_query']
            raw_steps_string = row['matched_blueprint_steps']

            # Safely parse the stringified list back into a real Python list
            try:
                steps_list = ast.literal_eval(raw_steps_string)
            except (ValueError, SyntaxError):
                steps_list = [raw_steps_string] # Fallback just in case

            # Map the raw query to the blueprint steps
            prefill_data[masked_query] = steps_list

    return prefill_data

if __name__ == "__main__":
    print("[SYSTEM] Loading cache prefill data from CSV...")
    prefill_dict = load_cache_from_csv("data/cache_telemetry_full.csv") 
    print(f"[SYSTEM] Successfully parsed {len(prefill_dict)} blueprints for prefilling.")

    print("[SYSTEM] Booting Financial Agent and warming vector cache...")
    agent = FinancialAgent(cache_prefill_info=prefill_dict)
    
    questions = get_questions(150)

    for i in range(141, 150):
        question = questions[i]
        print(f"\n=======================")
        print(f"Processing Question {i}")
        print(f"=======================")
        start_time = time.perf_counter()
        answer = agent.run(question, i)
        end_time = time.perf_counter()

        time_taken = round(end_time-start_time,2)
        print(f"\nQ: {question}\nA: {answer} in {time_taken:.3f} seconds")

        with open("data/time_stats.csv", "a") as csvfile:
            writer = csv.DictWriter(csvfile, ["question","answer","time_taken"])
            writer.writerow({"question":question, "answer": answer, "time_taken": time_taken})

