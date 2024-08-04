# MediTimely-AI
Logistic Regression ML for MediTimely App

# Hospital Data Analysis and AI Modeling

This repository contains scripts and data for analyzing hospital data and applying machine learning models to derive insights. The results are stored in Firebase.

## Contents

- `training data = sample_hospital_data_100.csv`
  - Sample data used for training the models.
  
- `actual data = hospital_doctors4.csv`
  - Actual hospital data used for prediction and analysis.

- `logistic_regression_model_hex.py`
  - A logistic regression model where AI results are saved in Firebase with a random UID as the parent node.

- `logistic_regression_model_id.py`
  - A logistic regression model where AI results are saved in Firebase with hospital IDs as the parent node.

- `random_forest_classifier_model.py`
  - A random forest classifier model as a variation of the logistic regression model.

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: `pyrebase`, `sklearn`, `pandas`, etc.
