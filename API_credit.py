# Inference server MLFlow + Fast API
# Return model class 0 and class 1 probabilities

from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
from typing import List

# On Anaconda command prompt :
# cd documents/python/projets/projet_7
# uvicorn API_credit:app --port 8000

# Create an FastAPI instance
app = FastAPI(title="Loan Default Prediction API")

# Chargement du modèle depuis le dossier local
model = mlflow.sklearn.load_model("model_heroku")

@app.get("/")
def root():
    return {"message": "API is running!!!"}   

class PredictRequest(BaseModel):
    data: List[List[float]]
    columns: List[str]

@app.post("/data_shape")
def predict_shape(request: PredictRequest):
    return {"message": "Received your request!", "data_shape": [len(request.data), len(request.columns)]}

@app.post("/predict")
def predict(request: PredictRequest):
    return {"message": "Received your request!", "data_shape": [len(request.data), len(request.columns)]}
