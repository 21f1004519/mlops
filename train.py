from feast import FeatureStore
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data from Feast
store = FeatureStore(repo_path="feature_repo")

training_df = store.get_historical_features(
    entity_df=pd.DataFrame({
        "iris_id": [1001, 1002, 1003],
        "event_timestamp": pd.Timestamp.now(),
    }),
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:species",
    ],
).to_df()

print("✅ Sample training data:\n", training_df.head())

# Prepare data
X = training_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = training_df["species"]

# Encode target labels (string → numeric)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"🎯 Model trained successfully — Accuracy: {acc:.2f}")

# Save model + label encoder
joblib.dump(clf, "models/iris_model_feast.joblib")
joblib.dump(le, "models/label_encoder.joblib")

# Materialize features into online store
store.materialize_incremental(end_date=pd.Timestamp.now())
print("✅ Features materialized to online store.")
