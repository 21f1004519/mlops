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
import random

# ---- CONFIG ----
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris_poison_experiment")

LOCAL_DATA_PATH = Path("data")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

data_path = LOCAL_DATA_PATH / "raw" / "iris.csv"
data = pd.read_csv(data_path)

# Poisoning levels (% of rows to corrupt)
POISON_LEVELS = [0, 0.05, 0.10, 0.50]  # 0% baseline, 5%, 10%, 50%

def poison_data(df, poison_ratio):
    df = df.copy()
    n = int(len(df) * poison_ratio)
    idx = np.random.choice(df.index, size=n, replace=False)

    # Inject random noise from uniform distribution (feature attack)
    df.loc[idx, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = np.random.uniform(
        low=df.min().min(), 
        high=df.max().max(), 
        size=(n, 4)
    )
    return df

for poison_ratio in POISON_LEVELS:
    poisoned_df = poison_data(data, poison_ratio)

    train, test = train_test_split(poisoned_df, test_size=0.4, stratify=poisoned_df['species'], random_state=42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train['species']
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']

    for depth in [2, 3, 4, 5]:
        with mlflow.start_run(run_name=f"poison_{int(poison_ratio*100)}pct_depth{depth}"):
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = metrics.accuracy_score(y_test, preds)

            mlflow.log_param("poison_ratio", poison_ratio)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)

            model_path = ARTIFACTS_DIR / f"model_poison{int(poison_ratio*100)}_depth{depth}.joblib"
            joblib.dump(model, model_path)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            print(f"Poison {poison_ratio*100}% | Depth {depth} | Accuracy: {acc:.4f}")

print("Poisoning experiment completed.")
