# Inference server MLFlow + Fast API
# Return model class 0 and class 1 probabilities

from fastapi import FastAPI, HTTPException
import mlflow
from pydantic import BaseModel
from typing import List

# On Anaconda command prompt :
# cd documents/python/projets/projet_7
# uvicorn API_credit:app --port 8000

# Create an FastAPI instance
app = FastAPI(title="Loan Default Prediction API")

@app.get("/")
def root():
    return {"message": "API is running!!!"}   

class PredictRequest(BaseModel):
    data: List[List[float]]
    columns: List[str]

@app.post("/predict")
def predict(request: PredictRequest):
    return {"message": "Received your request!", "data_shape": [len(request.data), len(request.columns)]}

