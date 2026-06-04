import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv("weather.csv")

    df.columns = df.columns.str.strip().str.lower()

    # Example target: rain (1 = rain, 0 = no rain)
    df = df.dropna()

    X = df.drop("rain", axis=1)
    y = df["rain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "weather_model.pkl")

    return model
