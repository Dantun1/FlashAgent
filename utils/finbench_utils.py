import os
import pandas as pd

def load_finbench():
    df = pd.read_json("hf://datasets/PatronusAI/financebench/financebench_merged.jsonl", lines=True)
    return df



def get_questions(quantity: int = None) -> list[str]:
    df = load_finbench()
    return df["question"].head(quantity).to_list()

def get_custom_questions(quantity: int = None) -> list[str]:
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "variable_tweak_data.csv")
    df = pd.read_csv(csv_path)
    questions = df["Question"].head(quantity).tolist()
    return questions

def get_evidence(row_index: int) -> str:
    df = load_finbench()
    return df.iloc[row_index]["evidence"]
