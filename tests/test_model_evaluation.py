import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Ensure it uses the correct MLflow server
mlflow.set_tracking_uri("http://34.10.27.138:5001")

def test_model_accuracy():
    df = pd.read_csv("data/raw/iris.csv")
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load latest registered model version from MLflow
    model_name = "IrisDecisionTreeModel"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc > 0.85, f"Model accuracy too low: {acc}"
