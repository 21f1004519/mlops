Assignment 1 – Week 1 – MLOps Course

Student ID: 21f1004519
Term: MAY 2025  

Assignment Objective:
Setting up the ML pipeline for the Iris Classifier in Google Cloud Platform (GCP) using Vertex AI and Google Cloud Storage (GCS).

---
Overview
This assignment demonstrates how to:
1. Activate and configure GCP services (Vertex AI, Cloud Storage, Compute Engine).
2. Store training datasets (`iris.csv`, `v1/data.csv`, `v2/data.csv`) in GCS.
3. Train a Decision Tree model on the Iris dataset inside Vertex AI Workbench.
4. Save model artifacts with timestamped folder names in GCS.
5. Run inference by downloading the trained model from GCS.
6. Execute training runs twice on the raw dataset, producing two artifact folders.
7. (Optional) Execute training on versioned datasets (`v1`, `v2`).

---

Files Included

1. Code/Scripts
- week1ga.ipynb  
  - Loads dataset from GCS (`raw`, `v1`, or `v2`).  
  - Trains a Decision Tree classifier.  
  - Evaluates accuracy.  
  - Saves model locally and uploads to GCS under a timestamped folder.  

- inference.ipynb  
  - Downloads the trained model from GCS.  
  - Performs predictions on evaluation data (`iris.csv`).  
  - Prints sample predictions and accuracy.  

2. README.md
- This file, explaining the utility of all included files and execution flow.  

---

Execution Instructions

Step 1: Environment Setup
- Enable APIs: Vertex AI, Cloud Storage, Compute Engine.  
- Create a Vertex AI Workbench instance.  

Step 2: Train the model 
- Run all the cells one by one in week1ga.ipynb
Each run uploads a model artifact into:
gs://<your-bucket>/artifacts/<version>/<timestamp>/model.joblib

Step 3: Run Inference
This downloads the model from GCS and prints predictions on the eval dataset.

Output Structure in GCS

Example layout after two runs on raw + one on v1 and v2:

gs://<project-id>-week1ga/
 ├── data/
 │    ├── raw/iris.csv
 │    ├── v1/data.csv
 │    └── v2/data.csv
 └── artifacts/
      ├── raw/2025-10-04_14-01-19/model.joblib
      ├── raw/2025-10-04_14-02-19//model.joblib
      ├── v1/2025-10-04_14-03-19//model.joblib
      └── v2/2025-10-04_14-04-19//model.joblib
---
Assignment 2 – Week 2 – MLOps Course

Student ID: 21f1004519
Term: MAY 2025  

Assignment Objective:
The goal of this assignment is to incorporate **DVC (Data Version Control)** into the Iris classification pipeline from Week 1.  
We track both datasets and trained model files, connect DVC to a **Google Cloud Storage (GCS) bucket** for remote storage, and demonstrate effortless version traversal using DVC.

---

## Repository Structure

├── data/
│ ├── raw/iris.csv # Original raw Iris dataset
│ ├── v1/data.csv # Version 1 dataset
│ └── v2/data.csv # Version 2 dataset
│── model.joblib # Model trained on data/raw/iris.csv dataset
│── model_v1.joblib # Model trained on data/v1/data.csv dataset
|── model_v2.joblib # Model trained on data/v2/v2/data.csv dataset
├── train.py # Training script for the Iris pipeline
├── inference.ipynb # Script to run inference
|── week1ga.ipynb # Week1 GA
├── README.md # This file
├── .dvc/ # DVC metadata folder
└── .gitignore # Git ignore file

---

## Steps Performed

### 1. Git & DVC Initialization
- Cloned the Week 1 repository locally.
- Initialized Git and DVC in the repo:
git init
dvc init
### 2. Configure GCS bucket as DVC remote:
dvc remote add -d myremote gs://heroic-throne-473405-m8-week2ga
git commit -m "Configured DVC remote on GCS"
### 3.Tracking Dataset and Model (Version 1):
- Added Iris dataset (data/raw/iris.csv) to DVC
dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc
git commit -m "Tracked dataset raw"
dvc push
### 4.Trained model using data/raw/iris.csv dataset:
python train.py --data data/raw/iris.csv --output model.joblib
### 5. Tracked trained model with DVC:
dvc add model.joblib
git add model.joblib.dvc
git commit -m "Model v1 trained on data/raw/iris.csv"
git tag -a "v1.0" -m "Model version 1.0 trained on data/raw/iris.csv data"
dvc push
### 6. Tracking Dataset and Model (Version 2)
- Added dataset (data/v1/data.csv) to DVC:
dvc add data/v1/iris.csv
git add data/v1/iris.csv.dvc
git commit -m "Tracked dataset v1"
dvc push
### 6. Trained model using data/v1/data.csv dataset:
python train.py --data data/v1/data.csv --output model_v2.joblib
### 7. Tracked trained model with DVC:
dvc add model_v2.joblib
git add model_v2.joblib.dvc
git commit -m "Model v2 trained on data/v1/data.csv"
git tag -a "v2.0" -m "Model version 2.0 trained on data/v1/data.csv data"
dvc push
### 8. Tracking Dataset and Model (Version 3)
- Added dataset (data/v2/data.csv) to DVC:
dvc add data/v2/iris.csv
git add data/v2/iris.csv.dvc
git commit -m "Tracked dataset v2"
dvc push
### 6. Trained model using data/v2/data.csv dataset:
python train.py --data data/v2/data.csv --output model_v2.joblib
### 7. Tracked trained model with DVC:
dvc add model_v2.joblib
git add model_v2.joblib.dvc
git commit -m "Model v2 trained on data/v2/data.csv"
git tag -a "v2.0" -m "Model version 2.0 trained on data/v2/data.csv data"
dvc push
### 8. Version Traversal using DVC
Switch to version 1:
git checkout v1.0
dvc checkout
ls -lart
- You will be able to see only the model.joblib and model.joblib.dvc files at the root of the repo and in the data folder, you will be able to see only the raw folder and its contents

Switch to version 2:

git checkout v2.0
dvc checkout
ls -lart
- You will be able to see the model_v1.joblib and model_v1.joblib.dvc files at the root of the repo now and in the data folder, you will be able to see the v1 folder and its contents now.

Switch to version 3:

git checkout v3.0
dvc checkout
ls -lart
- You will be able to see the model_v2.joblib and model_v2.joblib.dvc files at the root of the repo now and in the data folder, you will be able to see the v2 folder and its contents now.

DVC automatically restores the dataset and model corresponding to each version tag.

### 9.Google Cloud Storage Remote

All datasets and model artifacts are stored in GCS bucket:

gs://heroic-throne-473405-m8-week2ga

DVC ensures large files are not stored in Git, only in cloud remote.

### 10.Learnings

- Learned how to version-control large datasets and models using DVC.
- Learned to connect DVC with cloud storage (GCS) for scalable storage.
- Learned to traverse between data/model versions effortlessly with dvc checkout.
- Gained hands-on experience in reproducible ML workflows, which is critical in MLOps pipelines.

---