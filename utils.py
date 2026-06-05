import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def generate_insights(df):
    insights = []

    if df["temp"].mean() > 30:
        insights.append("High temperature detected 🌡️")
    else:
        insights.append("Normal temperature 🌤️")

    if df["humidity"].mean() > 70:
        insights.append("High humidity levels 💧")
    else:
        insights.append("Moderate humidity 👍")

    if df["rain"].sum() > len(df) * 0.5:
        insights.append("Frequent rain conditions 🌧️")
    else:
        insights.append("Low chance of rain ☀️")

    return insights
