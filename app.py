import streamlit as st
import pandas as pd
import joblib

from utils import load_data, generate_insights
from model import train_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Weather ML Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
df = load_data("weather.csv")

st.title("🌦️ Weather Forecast ML Dashboard")

# ---------------- AUTO DETECT TARGET COLUMN ----------------
possible_targets = ["rain", "Rain", "rainfall", "precipitation", "target"]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

# If no target column → create one
if target_col is None:
    st.warning("No target column found. Creating synthetic 'rain' column.")
    df["rain"] = (df["humidity"] > df["humidity"].median()).astype(int)
    target_col = "rain"
else:
    df.rename(columns={target_col: "rain"}, inplace=True)
    target_col = "rain"

# ---------------- MODEL ----------------
try:
    model = joblib.load("weather_model.pkl")
except:
    model = train_model(df)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input Features")

features = df.drop("rain", axis=1).columns

input_data = {}

for col in features:
    input_data[col] = st.sidebar.number_input(
        col,
        value=float(df[col].mean())
    )

input_df = pd.DataFrame([input_data])
input_df = input_df[features]

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Rain"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("🌧️ Rain Expected")
    else:
        st.success("☀️ No Rain Expected")

# ---------------- DATA PREVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- FILTER ----------------
st.subheader("Filter Data")

if "temp" in df.columns:
    min_temp = int(df["temp"].min())
    max_temp = int(df["temp"].max())

    temp_range = st.slider("Temperature Range", min_temp, max_temp, (min_temp, max_temp))

    filtered_df = df[(df["temp"] >= temp_range[0]) & (df["temp"] <= temp_range[1])]
else:
    filtered_df = df

st.dataframe(filtered_df)

# ---------------- INSIGHTS ----------------
st.subheader("AI Insights")

insights = generate_insights(filtered_df)

for i in insights:
    st.success(i)

st.caption("✔ Deployment Ready Streamlit App")
