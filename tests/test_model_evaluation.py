import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_model_accuracy():
    df = pd.read_csv("data/raw/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load("models/iris_model.joblib")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc > 0.85, f"Model accuracy too low: {acc}"
