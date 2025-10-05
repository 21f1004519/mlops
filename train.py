import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from google.cloud import storage
import joblib
import time
from datetime import datetime

DATA_VERSION = "v2"  # change to "raw", "v1", or "v2"
PROJECT_ID = "heroic-throne-473405-m8"
LOCATION = "us-central1"
BUCKET_URI = f"gs://heroic-throne-473405-m8-week1ga"
# Load data from GCS
data_path = f"{BUCKET_URI}/data/{DATA_VERSION}/data.csv" if DATA_VERSION != "raw" else f"{BUCKET_URI}/data/raw/iris.csv"
data = pd.read_csv(data_path)
print(data.head())

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

# Get the current time
timestamp = int(time.time())

# Convert to readable string (YYYY-MM-DD_HH-MM-SS)
timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")

joblib.dump(mod_dt, "model_v2.joblib")