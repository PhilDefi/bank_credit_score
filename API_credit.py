# Inference server MLFlow + Fast API
# Return model class 0 and class 1 probabilities

from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
from typing import List
import pandas as pd

import shap
import io
import base64
import matplotlib.pyplot as plt


# On Anaconda command prompt :
# cd documents/python/projets/projet_7
# uvicorn API_credit:app --port 8000

# Create an FastAPI instance
app = FastAPI(title="Loan Default Prediction API")

# Load model from model folder on GitHub (the GitHub repository is connected to Heroku)
model = mlflow.sklearn.load_model("model_LGBM_heroku")


dtypes_loaded = {'SK_ID_CURR': 'int64',
 'NAME_CONTRACT_TYPE': 'int32',
 'FLAG_OWN_CAR': 'int32',
 'FLAG_OWN_REALTY': 'int32',
 'CNT_CHILDREN': 'int64',
 'AMT_INCOME_TOTAL': 'float64',
 'AMT_CREDIT': 'float64',
 'AMT_ANNUITY': 'float64',
 'AMT_GOODS_PRICE': 'float64',
 'REGION_POPULATION_RELATIVE': 'float64',
 'DAYS_BIRTH': 'int64',
 'DAYS_EMPLOYED': 'float64',
 'DAYS_REGISTRATION': 'float64',
 'DAYS_ID_PUBLISH': 'int64',
 'OWN_CAR_AGE': 'float64',
 'FLAG_MOBIL': 'int64',
 'FLAG_EMP_PHONE': 'int64',
 'FLAG_WORK_PHONE': 'int64',
 'FLAG_CONT_MOBILE': 'int64',
 'FLAG_PHONE': 'int64',
 'FLAG_EMAIL': 'int64',
 'CNT_FAM_MEMBERS': 'float64',
 'REGION_RATING_CLIENT': 'int64',
 'REGION_RATING_CLIENT_W_CITY': 'int64',
 'HOUR_APPR_PROCESS_START': 'int64',
 'REG_REGION_NOT_LIVE_REGION': 'int64',
 'REG_REGION_NOT_WORK_REGION': 'int64',
 'LIVE_REGION_NOT_WORK_REGION': 'int64',
 'REG_CITY_NOT_LIVE_CITY': 'int64',
 'REG_CITY_NOT_WORK_CITY': 'int64',
 'LIVE_CITY_NOT_WORK_CITY': 'int64',
 'EXT_SOURCE_1_x': 'float64',
 'EXT_SOURCE_2_x': 'float64',
 'EXT_SOURCE_3_x': 'float64',
 'APARTMENTS_AVG': 'float64',
 'BASEMENTAREA_AVG': 'float64',
 'YEARS_BEGINEXPLUATATION_AVG': 'float64',
 'YEARS_BUILD_AVG': 'float64',
 'COMMONAREA_AVG': 'float64',
 'ELEVATORS_AVG': 'float64',
 'ENTRANCES_AVG': 'float64',
 'FLOORSMAX_AVG': 'float64',
 'FLOORSMIN_AVG': 'float64',
 'LANDAREA_AVG': 'float64',
 'LIVINGAPARTMENTS_AVG': 'float64',
 'LIVINGAREA_AVG': 'float64',
 'NONLIVINGAPARTMENTS_AVG': 'float64',
 'NONLIVINGAREA_AVG': 'float64',
 'APARTMENTS_MODE': 'float64',
 'BASEMENTAREA_MODE': 'float64',
 'YEARS_BEGINEXPLUATATION_MODE': 'float64',
 'YEARS_BUILD_MODE': 'float64',
 'COMMONAREA_MODE': 'float64',
 'ELEVATORS_MODE': 'float64',
 'ENTRANCES_MODE': 'float64',
 'FLOORSMAX_MODE': 'float64',
 'FLOORSMIN_MODE': 'float64',
 'LANDAREA_MODE': 'float64',
 'LIVINGAPARTMENTS_MODE': 'float64',
 'LIVINGAREA_MODE': 'float64',
 'NONLIVINGAPARTMENTS_MODE': 'float64',
 'NONLIVINGAREA_MODE': 'float64',
 'APARTMENTS_MEDI': 'float64',
 'BASEMENTAREA_MEDI': 'float64',
 'YEARS_BEGINEXPLUATATION_MEDI': 'float64',
 'YEARS_BUILD_MEDI': 'float64',
 'COMMONAREA_MEDI': 'float64',
 'ELEVATORS_MEDI': 'float64',
 'ENTRANCES_MEDI': 'float64',
 'FLOORSMAX_MEDI': 'float64',
 'FLOORSMIN_MEDI': 'float64',
 'LANDAREA_MEDI': 'float64',
 'LIVINGAPARTMENTS_MEDI': 'float64',
 'LIVINGAREA_MEDI': 'float64',
 'NONLIVINGAPARTMENTS_MEDI': 'float64',
 'NONLIVINGAREA_MEDI': 'float64',
 'TOTALAREA_MODE': 'float64',
 'OBS_30_CNT_SOCIAL_CIRCLE': 'float64',
 'DEF_30_CNT_SOCIAL_CIRCLE': 'float64',
 'OBS_60_CNT_SOCIAL_CIRCLE': 'float64',
 'DEF_60_CNT_SOCIAL_CIRCLE': 'float64',
 'DAYS_LAST_PHONE_CHANGE': 'float64',
 'FLAG_DOCUMENT_2': 'int64',
 'FLAG_DOCUMENT_3': 'int64',
 'FLAG_DOCUMENT_4': 'int64',
 'FLAG_DOCUMENT_5': 'int64',
 'FLAG_DOCUMENT_6': 'int64',
 'FLAG_DOCUMENT_7': 'int64',
 'FLAG_DOCUMENT_8': 'int64',
 'FLAG_DOCUMENT_9': 'int64',
 'FLAG_DOCUMENT_10': 'int64',
 'FLAG_DOCUMENT_11': 'int64',
 'FLAG_DOCUMENT_12': 'int64',
 'FLAG_DOCUMENT_13': 'int64',
 'FLAG_DOCUMENT_14': 'int64',
 'FLAG_DOCUMENT_15': 'int64',
 'FLAG_DOCUMENT_16': 'int64',
 'FLAG_DOCUMENT_17': 'int64',
 'FLAG_DOCUMENT_18': 'int64',
 'FLAG_DOCUMENT_19': 'int64',
 'FLAG_DOCUMENT_20': 'int64',
 'FLAG_DOCUMENT_21': 'int64',
 'AMT_REQ_CREDIT_BUREAU_HOUR': 'float64',
 'AMT_REQ_CREDIT_BUREAU_DAY': 'float64',
 'AMT_REQ_CREDIT_BUREAU_WEEK': 'float64',
 'AMT_REQ_CREDIT_BUREAU_MON': 'float64',
 'AMT_REQ_CREDIT_BUREAU_QRT': 'float64',
 'AMT_REQ_CREDIT_BUREAU_YEAR': 'float64',
 'CODE_GENDER_F': 'bool',
 'CODE_GENDER_M': 'bool',
 'CODE_GENDER_XNA': 'bool',
 'NAME_TYPE_SUITE_Children': 'bool',
 'NAME_TYPE_SUITE_Family': 'bool',
 'NAME_TYPE_SUITE_Group of people': 'bool',
 'NAME_TYPE_SUITE_Other_A': 'bool',
 'NAME_TYPE_SUITE_Other_B': 'bool',
 'NAME_TYPE_SUITE_Spouse, partner': 'bool',
 'NAME_TYPE_SUITE_Unaccompanied': 'bool',
 'NAME_INCOME_TYPE_Businessman': 'bool',
 'NAME_INCOME_TYPE_Commercial associate': 'bool',
 'NAME_INCOME_TYPE_Maternity leave': 'bool',
 'NAME_INCOME_TYPE_Pensioner': 'bool',
 'NAME_INCOME_TYPE_State servant': 'bool',
 'NAME_INCOME_TYPE_Student': 'bool',
 'NAME_INCOME_TYPE_Unemployed': 'bool',
 'NAME_INCOME_TYPE_Working': 'bool',
 'NAME_EDUCATION_TYPE_Academic degree': 'bool',
 'NAME_EDUCATION_TYPE_Higher education': 'bool',
 'NAME_EDUCATION_TYPE_Incomplete higher': 'bool',
 'NAME_EDUCATION_TYPE_Lower secondary': 'bool',
 'NAME_EDUCATION_TYPE_Secondary / secondary special': 'bool',
 'NAME_FAMILY_STATUS_Civil marriage': 'bool',
 'NAME_FAMILY_STATUS_Married': 'bool',
 'NAME_FAMILY_STATUS_Separated': 'bool',
 'NAME_FAMILY_STATUS_Single / not married': 'bool',
 'NAME_FAMILY_STATUS_Unknown': 'bool',
 'NAME_FAMILY_STATUS_Widow': 'bool',
 'NAME_HOUSING_TYPE_Co-op apartment': 'bool',
 'NAME_HOUSING_TYPE_House / apartment': 'bool',
 'NAME_HOUSING_TYPE_Municipal apartment': 'bool',
 'NAME_HOUSING_TYPE_Office apartment': 'bool',
 'NAME_HOUSING_TYPE_Rented apartment': 'bool',
 'NAME_HOUSING_TYPE_With parents': 'bool',
 'OCCUPATION_TYPE_Accountants': 'bool',
 'OCCUPATION_TYPE_Cleaning staff': 'bool',
 'OCCUPATION_TYPE_Cooking staff': 'bool',
 'OCCUPATION_TYPE_Core staff': 'bool',
 'OCCUPATION_TYPE_Drivers': 'bool',
 'OCCUPATION_TYPE_HR staff': 'bool',
 'OCCUPATION_TYPE_High skill tech staff': 'bool',
 'OCCUPATION_TYPE_IT staff': 'bool',
 'OCCUPATION_TYPE_Laborers': 'bool',
 'OCCUPATION_TYPE_Low-skill Laborers': 'bool',
 'OCCUPATION_TYPE_Managers': 'bool',
 'OCCUPATION_TYPE_Medicine staff': 'bool',
 'OCCUPATION_TYPE_Private service staff': 'bool',
 'OCCUPATION_TYPE_Realty agents': 'bool',
 'OCCUPATION_TYPE_Sales staff': 'bool',
 'OCCUPATION_TYPE_Secretaries': 'bool',
 'OCCUPATION_TYPE_Security staff': 'bool',
 'OCCUPATION_TYPE_Waiters/barmen staff': 'bool',
 'WEEKDAY_APPR_PROCESS_START_FRIDAY': 'bool',
 'WEEKDAY_APPR_PROCESS_START_MONDAY': 'bool',
 'WEEKDAY_APPR_PROCESS_START_SATURDAY': 'bool',
 'WEEKDAY_APPR_PROCESS_START_SUNDAY': 'bool',
 'WEEKDAY_APPR_PROCESS_START_THURSDAY': 'bool',
 'WEEKDAY_APPR_PROCESS_START_TUESDAY': 'bool',
 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY': 'bool',
 'ORGANIZATION_TYPE_Advertising': 'bool',
 'ORGANIZATION_TYPE_Agriculture': 'bool',
 'ORGANIZATION_TYPE_Bank': 'bool',
 'ORGANIZATION_TYPE_Business Entity Type 1': 'bool',
 'ORGANIZATION_TYPE_Business Entity Type 2': 'bool',
 'ORGANIZATION_TYPE_Business Entity Type 3': 'bool',
 'ORGANIZATION_TYPE_Cleaning': 'bool',
 'ORGANIZATION_TYPE_Construction': 'bool',
 'ORGANIZATION_TYPE_Culture': 'bool',
 'ORGANIZATION_TYPE_Electricity': 'bool',
 'ORGANIZATION_TYPE_Emergency': 'bool',
 'ORGANIZATION_TYPE_Government': 'bool',
 'ORGANIZATION_TYPE_Hotel': 'bool',
 'ORGANIZATION_TYPE_Housing': 'bool',
 'ORGANIZATION_TYPE_Industry: type 1': 'bool',
 'ORGANIZATION_TYPE_Industry: type 10': 'bool',
 'ORGANIZATION_TYPE_Industry: type 11': 'bool',
 'ORGANIZATION_TYPE_Industry: type 12': 'bool',
 'ORGANIZATION_TYPE_Industry: type 13': 'bool',
 'ORGANIZATION_TYPE_Industry: type 2': 'bool',
 'ORGANIZATION_TYPE_Industry: type 3': 'bool',
 'ORGANIZATION_TYPE_Industry: type 4': 'bool',
 'ORGANIZATION_TYPE_Industry: type 5': 'bool',
 'ORGANIZATION_TYPE_Industry: type 6': 'bool',
 'ORGANIZATION_TYPE_Industry: type 7': 'bool',
 'ORGANIZATION_TYPE_Industry: type 8': 'bool',
 'ORGANIZATION_TYPE_Industry: type 9': 'bool',
 'ORGANIZATION_TYPE_Insurance': 'bool',
 'ORGANIZATION_TYPE_Kindergarten': 'bool',
 'ORGANIZATION_TYPE_Legal Services': 'bool',
 'ORGANIZATION_TYPE_Medicine': 'bool',
 'ORGANIZATION_TYPE_Military': 'bool',
 'ORGANIZATION_TYPE_Mobile': 'bool',
 'ORGANIZATION_TYPE_Other': 'bool',
 'ORGANIZATION_TYPE_Police': 'bool',
 'ORGANIZATION_TYPE_Postal': 'bool',
 'ORGANIZATION_TYPE_Realtor': 'bool',
 'ORGANIZATION_TYPE_Religion': 'bool',
 'ORGANIZATION_TYPE_Restaurant': 'bool',
 'ORGANIZATION_TYPE_School': 'bool',
 'ORGANIZATION_TYPE_Security': 'bool',
 'ORGANIZATION_TYPE_Security Ministries': 'bool',
 'ORGANIZATION_TYPE_Self-employed': 'bool',
 'ORGANIZATION_TYPE_Services': 'bool',
 'ORGANIZATION_TYPE_Telecom': 'bool',
 'ORGANIZATION_TYPE_Trade: type 1': 'bool',
 'ORGANIZATION_TYPE_Trade: type 2': 'bool',
 'ORGANIZATION_TYPE_Trade: type 3': 'bool',
 'ORGANIZATION_TYPE_Trade: type 4': 'bool',
 'ORGANIZATION_TYPE_Trade: type 5': 'bool',
 'ORGANIZATION_TYPE_Trade: type 6': 'bool',
 'ORGANIZATION_TYPE_Trade: type 7': 'bool',
 'ORGANIZATION_TYPE_Transport: type 1': 'bool',
 'ORGANIZATION_TYPE_Transport: type 2': 'bool',
 'ORGANIZATION_TYPE_Transport: type 3': 'bool',
 'ORGANIZATION_TYPE_Transport: type 4': 'bool',
 'ORGANIZATION_TYPE_University': 'bool',
 'ORGANIZATION_TYPE_XNA': 'bool',
 'FONDKAPREMONT_MODE_not specified': 'bool',
 'FONDKAPREMONT_MODE_org spec account': 'bool',
 'FONDKAPREMONT_MODE_reg oper account': 'bool',
 'FONDKAPREMONT_MODE_reg oper spec account': 'bool',
 'HOUSETYPE_MODE_block of flats': 'bool',
 'HOUSETYPE_MODE_specific housing': 'bool',
 'HOUSETYPE_MODE_terraced house': 'bool',
 'WALLSMATERIAL_MODE_Block': 'bool',
 'WALLSMATERIAL_MODE_Mixed': 'bool',
 'WALLSMATERIAL_MODE_Monolithic': 'bool',
 'WALLSMATERIAL_MODE_Others': 'bool',
 'WALLSMATERIAL_MODE_Panel': 'bool',
 'WALLSMATERIAL_MODE_Stone, brick': 'bool',
 'WALLSMATERIAL_MODE_Wooden': 'bool',
 'EMERGENCYSTATE_MODE_No': 'bool',
 'EMERGENCYSTATE_MODE_Yes': 'bool',
 'DAYS_EMPLOYED_ANOM': 'bool',
 '1': 'float64',
 'EXT_SOURCE_1_y': 'float64',
 'EXT_SOURCE_2_y': 'float64',
 'EXT_SOURCE_3_y': 'float64',
 'EXT_SOURCE_1^2': 'float64',
 'EXT_SOURCE_1 EXT_SOURCE_2': 'float64',
 'EXT_SOURCE_1 EXT_SOURCE_3': 'float64',
 'EXT_SOURCE_1 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_2^2': 'float64',
 'EXT_SOURCE_2 EXT_SOURCE_3': 'float64',
 'EXT_SOURCE_2 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_3^2': 'float64',
 'EXT_SOURCE_3 DAYS_BIRTH': 'float64',
 'DAYS_BIRTH^2': 'float64',
 'EXT_SOURCE_1^3': 'float64',
 'EXT_SOURCE_1^2 EXT_SOURCE_2': 'float64',
 'EXT_SOURCE_1^2 EXT_SOURCE_3': 'float64',
 'EXT_SOURCE_1^2 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_1 EXT_SOURCE_2^2': 'float64',
 'EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3': 'float64',
 'EXT_SOURCE_1 EXT_SOURCE_2 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_1 EXT_SOURCE_3^2': 'float64',
 'EXT_SOURCE_1 EXT_SOURCE_3 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_1 DAYS_BIRTH^2': 'float64',
 'EXT_SOURCE_2^3': 'float64',
 'EXT_SOURCE_2^2 EXT_SOURCE_3': 'float64',
 'EXT_SOURCE_2^2 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_2 EXT_SOURCE_3^2': 'float64',
 'EXT_SOURCE_2 EXT_SOURCE_3 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_2 DAYS_BIRTH^2': 'float64',
 'EXT_SOURCE_3^3': 'float64',
 'EXT_SOURCE_3^2 DAYS_BIRTH': 'float64',
 'EXT_SOURCE_3 DAYS_BIRTH^2': 'float64',
 'DAYS_BIRTH^3': 'float64',
 'CREDIT_INCOME_PERCENT': 'float64',
 'ANNUITY_INCOME_PERCENT': 'float64',
 'CREDIT_TERM': 'float64',
 'DAYS_EMPLOYED_PERCENT': 'float64'}


# Test simple GET API
@app.get("/")
def root():
    return {"message": "API is running!!!"}   

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