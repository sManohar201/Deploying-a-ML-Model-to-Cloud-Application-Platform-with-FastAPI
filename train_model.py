"""
    Script to train machine learning model.
    Author:  Sabari Manohar
    Date:    March, 2024
"""

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import joblib

from ml.data import process_data  
from ml.model import train_model, compute_model_metrics, inference 

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# categorical feature hot coding
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def model_training():
    """
    This scripts loads the dataset and trains a classifier model.
    """
    # Data loading and preparation
    logger.info('Loading and splitting dataset')
    data_path =  'data/census_clean.csv'

    df = pd.read_csv(data_path)
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)  # Ensures reproducibility



    # Data preprocessing and model building
    logger.info('Preprocessing data and training model')
    X_train, y_train, encoder, lb = process_data(
                df_train, categorical_features=categorical_features, label="salary", training=True)

    X_test, y_test, _, _ = process_data(
                df_test, categorical_features=categorical_features, label="salary", 
                encoder=encoder, lb=lb, training=False)
    # training
    trained_model = train_model(X_train, y_train)

    # Model evaluation
    logger.info('Evaluating model performance')
    predictions = inference(model=trained_model, X=X_test)  # Using scikit-learn interface
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    logger.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

    # Serializing model artifacts
    logger.info('Saving model artifacts')
    joblib.dump(trained_model, 'model/model.pkl')
    joblib.dump(encoder, 'model/encoder.pkl')
    joblib.dump(lb, 'model/lb.pkl')

if __name__ == "__main__":
    logger.info("Run model training")
    model_training()
