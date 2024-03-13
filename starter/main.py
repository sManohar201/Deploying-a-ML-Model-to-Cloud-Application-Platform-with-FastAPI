import uvicorn
import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
from starter.ml.model_inference import predict  # More clarity
from starter.ml.data_processing import preprocess_data  # More clarity

# Define categorical features
categorical_features = [
    'work_category',  # Simpler names
    'education_level',
    'marital_status',
    'occupation_category',
    'relationship_status',
    'race',
    'gender',
    'country_origin'
]

# Input data model
class ClientInput(BaseModel):
    age: int = Field(examples=[29])
    work_category: str = Field(examples=['Union-gov'])
    final_weight: int = Field(examples=[82316])
    education_level: str = Field(examples=['Bachelors'])
    education_years: int = Field(alias='education-num', examples=[13])
    marital_status: str = Field(examples=['Never-married'])
    occupation_category: str = Field(examples=['Medicine'])
    relationship_status: str = Field(examples=['Bachelor'])
    race: str = Field(examples=['South-Asian'])
    gender: str = Field(examples=['Female'])
    capital_gain: int = Field(examples=[744])
    capital_loss: int = Field(examples=[32])
    weekly_work_hours: int = Field(alias='hours-per-week', examples=[50])
    country_origin: str = Field(alias='country-origin', examples=['India'])

# Paths to model artifacts
model_artifact_dir = 'model/'  # For consistent organization
model_file = model_artifact_dir + 'income_model.pkl' 
encoder_file = model_artifact_dir + 'data_encoder.pkl'
label_binarizer_file = model_artifact_dir + 'label_transformer.pkl'

# Load the artifacts
model = joblib.load(model_file)
encoder = joblib.load(encoder_file)
lb = joblib.load(label_binarizer_file)

# Initialize the FastAPI app
api = FastAPI()

@api.get('/')
async def welcome():
    return 'Welcome to the Income Prediction Service!'

@api.post('/prediction')
async def generate_prediction(client_input: ClientInput):
    input_df = pd.DataFrame(client_input.dict(by_alias=True), index=[0])

    prepared_data = preprocess_data(
        input_df, categorical_features=categorical_features, encoder=encoder, lb=lb
    )

    x_data, y_data, _ = prepared_data
    income_prediction = predict(model, x_data)

    predicted_category = lb.inverse_transform(income_prediction)[0]

    return {'Predicted Income': predicted_category}

if __name__ == "__main__":
    uvicorn.run('main:api', host='0.0.0.0', port=5000, reload=True)
