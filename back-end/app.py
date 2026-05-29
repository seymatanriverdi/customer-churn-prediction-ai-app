from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# APP
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL LOAD
model = joblib.load("churn_model.pkl")


# REQUEST MODEL
class CustomerData(BaseModel):
    gender_Male: int
    SeniorCitizen: int
    Partner_Yes: int
    Dependents_Yes: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

# HOME
@app.get("/")
def home():
    return {"message": "Customer Churn API Running"}

# PREDICT
@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "gender_Male": data.gender_Male,
        "SeniorCitizen": data.SeniorCitizen,
        "Partner_Yes": data.Partner_Yes,
        "Dependents_Yes": data.Dependents_Yes,
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges
    }])
    model_columns = joblib.load("model_columns.pkl")
    input_data = input_data.reindex(
    columns=model_columns,
    fill_value=0
)

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }

    
    