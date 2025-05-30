# Inference server MLFlow + Fast API
# Return model class 0 and class 1 probabilities

from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
import time

import shap
import io
import base64
import matplotlib.pyplot as plt


# On Anaconda command prompt :
# cd documents/python/projets/projet_7
# uvicorn API_credit2:app --port 8000
# rem : uvicorn can also be launched from Jupyter Lab terminal
# To be sure you're in the right virtual environment run (look at the star)
# conda info --envs (ou just conda info)
# rem 2 : to force uvicorn shutdown : Stop-Process -Name uvicorn

# Create an FastAPI instance
app = FastAPI(title="Loan Default Prediction API")

# Load model from model folder on GitHub (the GitHub repository is connected to Heroku)
model = mlflow.sklearn.load_model("model_LGBM_heroku")

# Get the features types from a .json file
with open("dtypes_enriched.json", "r") as file:
    dtypes_loaded = json.load(file)
dtypes_loaded.pop('TARGET', None)


# Test simple GET API
@app.get("/")
def root():
    return {"message": "API2 is running!!!"}   

@app.get("/info")
def api_info():
    return {
        "api_name": "Loan Default Prediction API",
        "api_version": "2",
        "model_name": "LightGBM Classifier",
        "training_dataset": "clean_dataset_final_v2.csv",
        "features_count": len(dtypes_loaded),
        "deployment_time": datetime.datetime.utcnow().isoformat() + "Z",
        "author": "PhilDefi"
    }


class PredictRequest(BaseModel):
    data: List[List[str]]
    columns: List[str]


# Test POST API that returns the shape of the input data
@app.post("/data_shape")
def predict_shape(request: PredictRequest):
    return {"message": "Received your request!", "data_shape": [len(request.data), len(request.columns)]}


# API that returns the prediction
@app.post("/predict")
async def predict(request: PredictRequest):
    # Print request content for checking
    print(request.data)
    print(request.columns)

    # Convert the input data to a pandas DataFrame
    df_post = pd.DataFrame(data=request.data, columns=request.columns)        
    
    # Convert columns to the specified data types
    for col, dtype in dtypes_loaded.items():
        if dtype == "bool":
            df_post[col] = df_post[col].map({'True': True, 'False': False})
        else:
            df_post[col] = df_post[col].astype(dtype)

    # NB : can not apply directly astype method for boolean variables
    # => return true if non-empty. Thus 'False' and 'True' strings are converted to True
    
    print("\ndtypes verification :\n", df_post.dtypes[0:20])
    print('**************************************************')
    print('df_post :\n', df_post)
    
    # Make predictions using the MLflow model
    predictions = model.predict_proba(df_post)
    print('Prediction made : ',predictions)
    
    # Return the predictions in the response
    predictions_list = predictions.tolist()
    return {"predictions": predictions_list}


# API that returns the prediction together with the SHAP interpretability, as a waterfall plot image
@app.post("/predict_with_explanation")
async def predict_with_explanation(request: PredictRequest):
    # Convert the input data to a pandas DataFrame
    df_post = pd.DataFrame(data=request.data, columns=request.columns)        
    
    # Convert columns to the specified data types
    for col, dtype in dtypes_loaded.items():
        if dtype == "bool":
            df_post[col] = df_post[col].map({'True': True, 'False': False})
        else:
            df_post[col] = df_post[col].astype(dtype)

    # Get preprocessing and model from pipeline
    preprocess = model.named_steps['pipeline']  # from make_pipeline
    classifier = model.named_steps['lgbmclassifier']

    # Preprocess input
    df_post_processed = preprocess.transform(df_post)

    # Predict probability
    probability = classifier.predict_proba(df_post_processed)[0, 1]

    # SHAP local explanation
    explainer = shap.Explainer(classifier)
    shap_values = explainer(df_post_processed)
    shap_values.feature_names = df_post.columns.tolist()
    print("SHAP VALUES : ", shap_values)

    # Generate waterfall plot
    shap.plots.waterfall(shap_values[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    shap_img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "probability_default": probability,
        "shap_waterfall_plot": shap_img_base64
    }