from feast import FeatureStore
import joblib
import pandas as pd

store = FeatureStore(repo_path="feature_repo")
model = joblib.load("models/iris_model_feast.joblib")
le = joblib.load("models/label_encoder.joblib")

# Fetch features for a given iris_id
feature_vector = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=[{"iris_id": 1001}],
).to_dict()

X = pd.DataFrame(feature_vector)
print("📘 Fetched features from online store:\n", X)

# Predict species
pred = model.predict(X)
decoded_species = le.inverse_transform(pred)
print(f"🌸 Predicted species: {decoded_species[0]}")