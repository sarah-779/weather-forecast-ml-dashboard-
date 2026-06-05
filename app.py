import streamlit as st
import pandas as pd
import joblib
from utils import load_data, generate_insights
from model import train_model

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Weather App", layout="wide")

# ---------------- LOAD DATA ----------------
df = load_data("weather.csv")

df.columns = df.columns.str.strip().str.lower()

# ---------------- CREATE TARGET SAFELY ----------------
if "humidity" in df.columns:
    base = df["humidity"]
elif "hum" in df.columns:
    base = df["hum"]
elif "humidity(%)" in df.columns:
    base = df["humidity(%)"]
else:
    st.error("No humidity column found")
    st.stop()

df["rain"] = (base > base.median()).astype(int)

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
