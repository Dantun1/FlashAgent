import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def load_finbench():
    df = pd.read_json("hf://datasets/PatronusAI/financebench/financebench_merged.jsonl", lines=True)
    return df



def get_questions(quantity: int = None) -> list[str]:
    df = load_finbench()
    return df["question"].head(quantity).to_list()

def get_custom_questions(quantity: int = None) -> list[str]:
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", os.getenv("CUSTOM_DATASET"))
    df = pd.read_csv(csv_path)
    questions = df["Question"].head(quantity).tolist()
    return questions

def get_evidence(row_index: int) -> str:
    df = load_finbench()
    return df.iloc[row_index]["evidence"]
