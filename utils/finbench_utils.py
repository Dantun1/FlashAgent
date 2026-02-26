import pandas as pd

def load_finbench():
    df = pd.read_json("hf://datasets/PatronusAI/financebench/financebench_merged.jsonl", lines=True)
    return df



def get_questions():
    df = load_finbench()
    return df["question"].to_list()
