from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load saved model
model = joblib.load("xgb_model.pkl")

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[:,1][0]
    pred = int(proba >= 0.84)  # using threshold
    return {"probability": proba, "prediction": pred}
