import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from utils import load_data, generate_insights
from model import train_model

st.set_page_config(page_title="Weather ML Dashboard", layout="wide")

# Load data
df = load_data("weather.csv")

# Load or train model
try:
    model = joblib.load("weather_model.pkl")
except:
    model = train_model()

st.title("🌦️ Weather Forecast ML Dashboard")
st.markdown("Deep Analytics + Machine Learning Weather Prediction System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Weather Input")

input_data = {}

for col in df.drop("rain", axis=1).columns:
    input_data[col] = st.sidebar.number_input(col)

input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict Rain"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("🌧️ Rain Expected")
    else:
        st.success("☀️ No Rain Expected")

# ---------------- FILTER ----------------
st.sidebar.header("Filters")

min_temp = int(df["temp"].min())
max_temp = int(df["temp"].max())

temp_range = st.sidebar.slider(
    "Temperature Range",
    min_temp,
    max_temp,
    (min_temp, max_temp)
)

filtered_df = df[
    (df["temp"] >= temp_range[0]) &
    (df["temp"] <= temp_range[1])
]

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Charts", "🧠 Insights"])

# OVERVIEW
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Records", len(filtered_df))
    c2.metric("Avg Temp", round(filtered_df["temp"].mean(), 2))
    c3.metric("Rain Cases", int(filtered_df["rain"].sum()))

    st.dataframe(filtered_df.head())

# CHARTS
with tab2:
    st.subheader("Weather Analytics")

    fig1 = px.line(filtered_df, y="temp", title="Temperature Trend")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(filtered_df, x="humidity", color="rain",
                  title="Humidity vs Rain")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(filtered_df, x="temp", y="humidity",
                      color="rain", title="Temp vs Humidity")
    st.plotly_chart(fig3, use_container_width=True)

# INSIGHTS
with tab3:
    st.subheader("AI Insights")

    insights = generate_insights(filtered_df)

    for i in insights:
        st.success(i)

    st.warning("This is for educational purposes only.")
