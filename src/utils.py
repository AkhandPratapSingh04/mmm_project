import pandas as pd
import os

def read_excel(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
