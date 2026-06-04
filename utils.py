import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df


def generate_insights(df):
    insights = []

    if df["temp"].mean() > 30:
        insights.append("High average temperature detected.")

    if df["humidity"].mean() > 70:
        insights.append("High humidity levels may indicate rainfall chances.")

    if df["rain"].mean() > 0.3:
        insights.append("Frequent rainfall patterns observed.")

    insights.append("Weather conditions are strongly seasonal.")

    return insights
