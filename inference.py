from feast import FeatureStore
import joblib
import pandas as pd

# Initialize Feast store and load model
store = FeatureStore(repo_path="feature_repo")
model = joblib.load("models/iris_model_feast.joblib")
le = joblib.load("models/label_encoder.joblib")

# Fetch features for a given iris_id
entity_rows = [{"iris_id": 1001}]

# Retrieve feature vector from online store
feature_vector = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=entity_rows,
).to_df()

print("📘 Raw features fetched from online store:\n", feature_vector)

# Keep only model features (drop iris_id if present)
X = feature_vector[
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
]

print("\n✅ Cleaned feature vector for prediction:\n", X)

# Predict and decode label
pred = model.predict(X)
decoded_species = le.inverse_transform(pred)
print(f"\n🌸 Predicted species: {decoded_species[0]}")
