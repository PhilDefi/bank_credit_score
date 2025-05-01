# Inference server MLFlow + Fast API
# Return model class 0 and class 1 probabilities

from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import mlflow

# On Anaconda command prompt :
# cd documents/python/projets/projet_7
# uvicorn API_credit:app --port 8000

# Create an FastAPI instance
app = FastAPI(title="Loan Default Prediction API")

@app.get("/")
def root():
    return {"message": "API is running!!!"}   

