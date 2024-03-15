"""
    This script provide FASTapi endpoints.
    Author: Sabari Manohar
    Date:   March, 2024
"""

import os
import joblib
import logging
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from ml.model import inference # More clarity
from ml.data import process_data  # More clarity

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Define categorical features
categorical_features = [
    'workclass',  
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")

# Input data model
class ClientInput(BaseModel):
    age: int = Field(..., example=50)
    workclass: str = Field(..., example="Private") 
    fnlgt: int = Field(..., example=234721) 
    education: str = Field(..., example="Doctorate")
    education_num: int = Field(..., example=16)
    marital_status: str = Field(..., example="Separated")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")

    class ConfigDict:
        json_schema_extra = {
                        "example": {
                                    'age':50,
                                    'workclass':"Private", 
                                    'fnlgt':234721,
                                    'education':"Doctorate",
                                    'education_num':16,
                                    'marital_status':"Separated",
                                    'occupation':"Exec-managerial",
                                    'relationship':"Not-in-family",
                                    'race':"Black",
                                    'sex':"Female",
                                    'capital_gain':0,
                                    'capital_loss':0,
                                    'hours_per_week':50,
                                    'native_country':"United-States"
                                    }
                        }

# Paths to model artifacts
model_artifact_dir = 'model/'  # For consistent organization
model_file = model_artifact_dir + 'model.pkl' 
encoder_file = model_artifact_dir + 'encoder.pkl'
label_binarizer_file = model_artifact_dir + 'lb.pkl'


api = FastAPI(title="Prediction API", 
              description="Runs prediction on a sample.",
              version="1.0.0")


@api.on_event("startup")
async def load_model():
    global model, encoder, lb
    global model_file, encoder_file, label_binarizer_file
    model = joblib.load(model_file)
    encoder = joblib.load(encoder_file)
    lb = joblib.load(label_binarizer_file)

@api.on_event("shutdown")
async def clear_model():
    model.clear()
    encoder.clear()
    lb.clear()

@api.get('/')
async def welcome():
    return 'Welcome to the Income Prediction Service!'

@api.post('/predict')
async def generate_prediction(client_input: ClientInput):
    # create dictionary
    data_old = client_input.dict()
    data = {}
    for key, value in data_old.items():
        new_key = hyphen_to_underscore(key)
        data[new_key] = value

    input_df = pd.DataFrame(data, index=[0])

    if os.path.isfile(model_file):
        model = joblib.load(model_file)
        encoder = joblib.load(encoder_file)
        lb = joblib.load(label_binarizer_file)

    prepared_data = process_data(
        input_df, categorical_features=categorical_features, training=False, 
        encoder=encoder, lb=lb
    )
    
    x_data, y_data, _, _ = prepared_data
    income_prediction = inference(model, x_data)

    if income_prediction[0] > 0.5:
        income_prediction = '>50K'
    else:
        income_prediction = '<=50K'
    data['prediction'] = income_prediction

    return data 