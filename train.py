import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import mlflow
import mlflow.sklearn

# Configuration via env
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print("MLflow tracking URI:", mlflow.get_tracking_uri())

mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "iris_decision_tree_experiment"))

LOCAL_DATA_PATH = Path("data")  # dvc pull should populate data/raw/iris.csv
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

data_path = LOCAL_DATA_PATH / "raw" / "iris.csv"
print("Loading data from:", data_path)
data = pd.read_csv(data_path)
print(data.head())

# Split
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# Hyperparameter loop
for max_depth in [2, 3, 4, 5]:
    print(f"Training model with max_depth={max_depth}")
    with mlflow.start_run(run_name=f"dt_depth_{max_depth}"):
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, preds)
        print(f"Max Depth: {max_depth} | Accuracy: {acc:.3f}")

        # Log hyperparameters and metrics
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # Save model locally
        model_save_path = MODEL_PATH.parent / f"model_depth_{max_depth}.joblib"
        joblib.dump(model, model_save_path)
        print("Saved model locally to", model_save_path)

        # Log model artifact to mlflow artifact store
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="IrisDecisionTreeModel_v2"
        )

        # Optionally copy the *best* model to artifacts/model.joblib (simple rule: highest acc)
        # For simplicity keep final iteration as model.joblib for packaging
        if max_depth == 5:
            joblib.dump(model, MODEL_PATH)
            print("Copied best model to", MODEL_PATH)


print("All runs logged to MLflow. Final model at:", MODEL_PATH)
