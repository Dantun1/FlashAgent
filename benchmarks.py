import argparse
import csv
import os
import time
import json

from plan_cache_engine import PlanCacheEngine
from utils.finbench_utils import get_questions

def run_evaluation(test_queries, output_csv="./data/cache_telemetry.csv", cache_enabled=True):
    engine = PlanCacheEngine(cache_enabled=cache_enabled)

    fieldnames = [
        "query_index", "original_query", "query_classification", "key_query", 
        "matched_blueprint_tag", "matched_blueprint_steps",
        "cosine_similarity", "cache_status", 
        "latency_ms", 
        "vertex_input_tokens", "vertex_output_tokens",
        "is_correct_routing"
    ]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for idx, query in enumerate(test_queries):
        start_time = time.perf_counter()
        
        result = engine.retrieve_plan(query)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Log the results
        row_data = {
            "query_index": idx + 1,
            "original_query": query,
            "query_classification": result["tag"].split("]")[0] + "]",
            "key_query": result["tag"],
            "matched_blueprint_tag": result["blueprint"].tag,
            "matched_blueprint_steps": result["blueprint"].steps,
            "cosine_similarity": round(result["score"], 4),
            "cache_status": result["status"],
            "latency_ms": round(latency_ms, 2),
            "vertex_input_tokens": result["inp_tokens"],    
            "vertex_output_tokens": result["out_tokens"],  
            "is_correct_routing": "" # to be added manually (false/true positives)
        }
        
        with open(output_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
            
        print(f"[{result['status']}] {latency_ms:.2f}ms | Score: {result['score']:.3f} | {query[:40]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run plan cache evaluation benchmark.")
    parser.add_argument("--baseline", action="store_true", help="Disable cache (baseline mode: every query pays full generation cost).")
    parser.add_argument("--output", default="./data/cache_telemetry.csv", help="Path for the output CSV.")
    parser.add_argument("--n", type=int, default=150, help="Number of queries to evaluate.")
    args = parser.parse_args()

    questions = get_questions(quantity=args.n)
    run_evaluation(questions, output_csv=args.output, cache_enabled=not args.baseline)