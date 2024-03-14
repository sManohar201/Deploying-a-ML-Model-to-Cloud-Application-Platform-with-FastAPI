"""
    This script provide FASTapi endpoints.
    Author: Sabari Manohar
    Date:   March, 2024
"""

import uvicorn
import os
import joblib
import pickle
import logging
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
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

# Input data model
class ClientInput(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
                        "example": {"age": 52,
                                    "workclass": "Self-emp-inc",
                                    "fnlgt": 287927,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Wife",
                "race": "White",
                "sex": "Female",
                "capital_gain": 15024,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
                                    }
                        }

# Paths to model artifacts
model_artifact_dir = 'model/'  # For consistent organization
model_file = model_artifact_dir + 'model.pkl' 
encoder_file = model_artifact_dir + 'encoder.pkl'
label_binarizer_file = model_artifact_dir + 'lb.pkl'

model = joblib.load(model_file)
encoder = joblib.load(encoder_file)
lb = joblib.load(label_binarizer_file)

# Initialize the FastAPI app
api = FastAPI(title="Prediction API", 
              description="Runs prediction on a sample.",
              version="1.0.0")


@api.get('/')
async def welcome():
    return 'Welcome to the Income Prediction Service!'

@api.post('/dummy')
async def check(input: ClientInput):
    print(type(input))
    print(input.age)
    return int(input.age)

@api.post('/predict')
async def generate_prediction(client_input: ClientInput):
    logger.info(f"{client_input.age}")
    # create dictionary
    data = {'age': client_input.age,
                'workclass': client_input.workclass, 
                'fnlgt': client_input.fnlgt,
                'education': client_input.education,
                'education-num': client_input.education_num,
                'marital-status': client_input.marital_status,
                'occupation': client_input.occupation,
                'relationship': client_input.relationship,
                'race': client_input.race,
                'sex': client_input.sex,
                'capital-gain': client_input.capital_gain,
                'capital-loss': client_input.capital_loss,
                'hours-per-week': client_input.hours_per_week,
                'native-country': client_input.native_country,
                }
    input_df = pd.DataFrame(data, index=[0])

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

if __name__ == "__main__":
    # uvicorn.run('main:api', host='0.0.0.0', port=5000, reload=True)
    pass
