# Inference server MLFlow + Fast API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
import pandas as pd
import mlflow.pyfunc
import mlflow
import json

# On Anaconda command prompt :
# cd documents/python/projets/projet_7
# uvicorn API_credit:app --port 8000

# browser : http://127.0.0.1:8000

# Create an FastAPI instance
app = FastAPI(title="Loan Default Prediction API")

# Define MLFlow tracking server location
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load best model from MLflow
model_uri = "runs:/e46c6c3acae346f5b7f2189a3d35def3/Master_Model"
model = mlflow.pyfunc.load_model(model_uri)


########################################################################
# Get saved data from /data folder
# Load variable type of main dataset
with open('./data/dtypes_enriched.json', 'r') as f:
    dtypes_loaded = json.load(f)
    
# Load CSV file forcing correct variable type for test purpose only
df2 = pd.read_csv('./data/train_enriched.csv', dtype=dtypes_loaded)

# Remove the target
X = df2.drop(columns=['TARGET'])
dtypes_loaded.pop('TARGET', None)
#########################################################################


@app.get("/")
def root():
    return {"message": "API is running!!!"}
    


class PredictRequest(BaseModel):
    data: List[List[str]]
    columns: List[str]

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
    predictions = model.predict(df_post)
    predictions = model.predict_proba(df_post)
    print('Prediction made : ',predictions) 
    
    # Convert predictions to a list or any other format you need
    predictions_list = predictions.tolist()

    # Return the predictions in the response
    return {"predictions": predictions_list}
