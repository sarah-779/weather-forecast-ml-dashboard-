import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):

    # ---------------- KEEP ONLY NUMERIC COLUMNS ----------------
    df = df.copy()

    X = df.drop("rain", axis=1)
    y = df["rain"]

    # convert everything to numeric (VERY IMPORTANT)
    X = X.select_dtypes(include=["number"])

    # handle missing values
    X = X.fillna(X.mean())

    # safety check
    if X.shape[1] == 0:
        raise Exception("No numeric features available for training")

    # ---------------- TRAIN MODEL ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "weather_model.pkl")

    return model
