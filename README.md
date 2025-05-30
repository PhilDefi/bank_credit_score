# 🏦 Credit Loan Repayment Prediction

A full-stack data science project to predict whether a bank loan will be repaid, using a trained machine learning model served via an API. Users can interact with the model through a user-friendly web interface built with Streamlit.

---

## 🚀 Project Overview

This project uses a machine learning model trained on credit application data to predict the probability of loan repayment. It provides:

- A RESTful API to serve predictions (`FastAPI`, deployed on Heroku)
- A web interface for CSV upload and prediction (`Streamlit`)
- Versioned model management using `MLflow`

---

## 🧠 Model Pipeline

The model was trained on historical loan data with features such as:
- Applicant income
- Loan amount
- Credit history
- Employment status
- ...

Pipeline includes:
- Preprocessing (imputation, encoding, scaling)
- Class imbalance handling (e.g., SMOTE)
- Model training (XGBoost, AdaBoost, LightGBM)
- Evaluation (F1-score, customized score, ROC AUC, SHAP explainability)

---

## ! Files and Folder organisation
- Project7V2.ipynb : main Jupyter notebook
- API_credit.py : API for requesting model on the Heroku server
- streamlit_interface.py : Streamlit end-user interface for credit prediction
- /model /model_LGBM_Heroku : folder containing previous prediction models
- /model_heroku : folder containing the final prediction model
- .github/workflows/ : GitHubActions folder, including .yml configuration file
