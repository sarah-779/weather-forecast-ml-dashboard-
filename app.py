import streamlit as st
import pandas as pd
import joblib
from utils import load_data, generate_insights
from model import train_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Weather App", layout="wide")

# ---------------- LOAD DATA ----------------
df = load_data("weather.csv")

# normalize column names
df.columns = df.columns.str.strip().str.lower()

# ---------------- HANDLE HUMIDITY SAFELY ----------------
humidity_col = None

possible_humidity_cols = ["humidity", "hum", "humidity(%)", "rh"]

for col in possible_humidity_cols:
    if col in df.columns:
        humidity_col = col
        break

# if no humidity column found → create fake one
if humidity_col is None:
    st.warning("No humidity column found. Creating synthetic humidity.")
    df["humidity"] = df.select_dtypes(include="number").mean(axis=1)
    humidity_col = "humidity"

# ---------------- CREATE TARGET ----------------
df["rain"] = (df[humidity_col] > df[humidity_col].median()).astype(int)

# ---------------- MODEL ----------------
try:
    model = joblib.load("weather_model.pkl")
except:
    model = train_model(df)

# ---------------- TITLE ----------------
st.title("🌦️ Weather ML App")

# ---------------- INPUT ----------------
features = df.drop("rain", axis=1).columns

input_data = {}

for col in features:
    input_data[col] = st.number_input(col, float(df[col].mean()))

input_df = pd.DataFrame([input_data])

# ---------------- PREDICT ----------------
if st.button("Predict"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("🌧️ Rain Expected")
    else:
        st.success("☀️ No Rain Expected")

# ---------------- INSIGHTS ----------------
st.subheader("Insights")

for i in generate_insights(df):
    st.success(i)
