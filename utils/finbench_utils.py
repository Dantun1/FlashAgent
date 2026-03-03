import os
import pandas as pd

def load_finbench():
    df = pd.read_json("hf://datasets/PatronusAI/financebench/financebench_merged.jsonl", lines=True)
    return df



def get_questions(quantity: int):
    df = load_finbench()
    return df["question"].head(quantity).to_list()

def get_custom_questions():
    xlsx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "custom_dataset.xlsx")
    df = pd.read_excel(xlsx_path)
    questions = df.loc[df["Variation Type"].notna(), "Question"].tolist()
    return questions

def get_evidence(row_index: int) -> str:
    df = load_finbench()
    return df.iloc[row_index]["evidence"]
