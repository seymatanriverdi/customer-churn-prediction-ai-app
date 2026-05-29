## Features
- Customer churn prediction using Machine Learning
- XGBoost model with hyperparameter tuning
- FastAPI backend
- React frontend
- Churn probability scoring
- Risk classification (Low / Medium / High)

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

### Backend
```bash
cd back-end
uvicorn app:app --reload
---

## How to Run Frontend
```md
### Frontend
```bash
cd front-end/churn-frontend
npm install
npm run dev
---
## API Documentation
```md
### Swagger UI
After starting the backend:
http://127.0.0.1:8000/docs
-----

customer-churn-ai-project/
│
├── back-end/
│
├── front-end/
│   └── churn-frontend/
│
├── training/
│
└── README.md
