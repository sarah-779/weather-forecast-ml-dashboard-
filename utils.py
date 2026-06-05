import pandas as pd

def load_data(path):
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    return df


def generate_insights(df):
    insights = []

    if "temp" in df.columns and df["temp"].mean() > 30:
        insights.append("High temperature detected 🌡️")
    else:
        insights.append("Normal temperature 🌤️")

    if "humidity" in df.columns and df["humidity"].mean() > 70:
        insights.append("High humidity 💧")
    else:
        insights.append("Normal humidity 👍")

    if "rain" in df.columns:
        if df["rain"].sum() > len(df) * 0.5:
            insights.append("High chance of rain 🌧️")
        else:
            insights.append("Low chance of rain ☀️")

    return insights
