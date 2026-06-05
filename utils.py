import pandas as pd

def load_data(path):
    df = pd.read_csv(path, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df
