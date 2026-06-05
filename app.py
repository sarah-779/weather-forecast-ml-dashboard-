import streamlit as st
import pandas as pd
import joblib
from utils import load_data, generate_insights
from model import train_model

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Weather ML App", layout="wide")

# ---------------- LOAD DATA ----------------
df = load_data("weather.csv")

# safety check
if "rain" not in df.columns:
    st.error("Dataset must contain 'rain' column (0/1)")
    st.stop()

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("weather_model.pkl")
except:
    model = train_model(df)

# ---------------- TITLE ----------------
st.title("🌦️ Weather Prediction App")
st.write("Simple ML model to predict rain/no rain")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Enter Weather Data")

features = df.drop("rain", axis=1).columns

user_input = {}

for col in features:
    user_input[col] = st.sidebar.number_input(
        col,
        value=float(df[col].mean())
    )

input_df = pd.DataFrame([user_input])
input_df = input_df[features]

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict"):
    result = model.predict(input_df)[0]

    if result == 1:
        st.error("🌧️ Rain Expected")
    else:
        st.success("☀️ No Rain Expected")

# ---------------- DATA VIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- FILTER ----------------
st.subheader("Filter by Temperature")

min_temp = int(df["temp"].min())
max_temp = int(df["temp"].max())

temp_range = st.slider("Temperature Range", min_temp, max_temp, (min_temp, max_temp))

filtered_df = df[(df["temp"] >= temp_range[0]) & (df["temp"] <= temp_range[1])]

st.write("Filtered Data")
st.dataframe(filtered_df)

# ---------------- INSIGHTS ----------------
st.subheader("AI Insights")

insights = generate_insights(filtered_df)

for i in insights:
    st.success(i)

st.caption("Built using Streamlit + Machine Learning")
