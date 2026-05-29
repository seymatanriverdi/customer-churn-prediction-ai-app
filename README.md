# Customer Churn Prediction AI App

An end-to-end machine learning application that predicts customer churn risk using a trained XGBoost model, FastAPI backend, and React frontend.

## Project Overview

This project predicts whether a customer is likely to churn based on customer profile, contract, payment, and billing information.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- React
- Vite
- Joblib

## ML Workflow

- Data loading
- Data cleaning
- Handling TotalCharges type issue
- Categorical encoding with get_dummies
- Train/test split
- Model comparison
- Hyperparameter tuning with RandomizedSearchCV
- ROC-AUC evaluation
- Feature importance analysis
- Model saving with Joblib

## Best Model

Tuned XGBoost Classifier

## Key Metrics

- ROC-AUC: 0.84
- Churn recall improved after tuning

## Important Features

- Contract_One year
- Contract_Two year
- InternetService_Fiber optic
- PaymentMethod_Electronic check

## API Endpoint

POST /predict

Example request:

```json
{
  "gender_Male": 1,
  "SeniorCitizen": 0,
  "Partner_Yes": 1,
  "Dependents_Yes": 0,
  "tenure": 24,
  "MonthlyCharges": 70,
  "TotalCharges": 1500
}