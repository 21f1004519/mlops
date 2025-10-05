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
