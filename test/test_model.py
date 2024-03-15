"""
    This script contains pytest modules for the model.
    Author: Sabari Manohar 
    Date: March, 2024 
"""

import sys
sys.path.append('.')  # Ensure project modules are in search path
import joblib
import pytest
import logging
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from ml.data import process_data  # Adjusted module name
from ml.model import inference, train_model

logging.getLogger(__name__)

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

def test_model_loading():
    """Verifies that the saved model can be loaded and is of the expected type.
    """

    try:
        loaded_model = joblib.load('model/model.pkl')  
    except FileNotFoundError as e:
        logging.error("the model file is not found.")
        raise e
    try:
        data_df = pd.read_csv('data/census_clean.csv')
    except FileNotFoundError as e:
        logging.error("the data file is not found in the path.")
        raise e
    
    x_trian, _, _, _ = process_data(
        data_df, categorical_features, label="salary", training=True
    )

    try:
        assert isinstance(loaded_model, RandomForestClassifier)
    except AssertionError as e:
        logging.error("The model is not a random forest classifier.")
        raise e
    try:
        loaded_model.predict(x_trian)
    except NotFittedError as e:
        logging.error("the model is not fitted. Train your model.")
        raise e
    pytest.df = data_df

def test_data_integrity():
    """Checks if the dataset can be loaded and contains at least one data point.
    """
    df = pytest.df 
    try:
        assert df.shape[0] > 0  # Ensure there are rows
    except AssertionError as e:
        logging.error("The data has no rows.")
        raise e

def test_processing_consistency():
    """Ensures that data processing produces feature and target arrays with compatible shapes.
    """
    df = pytest.df
    df_train, _ = train_test_split(df, test_size=0.20, random_state=42) 

    X_train, y_train, _, _ = process_data(
        df_train, categorical_features, label="salary", training=True
    )
    try:
        assert X_train.shape[0] == y_train.shape[0]
    except AssertionError as e:
        logging.error("The processed data has uneven feature rows and target rows.")
        raise e

def test_inference():
    """Ensures the training pipeline runs as expected and inference is right.
    """
    x_dummy = np.random.rand(30, 5)
    y_dummy = np.random.randint(2, size=30)

    model_dummy = train_model(x_dummy, y_dummy)
    try:
        y_prediction = inference(model_dummy, x_dummy)
    except NotFittedError as e:
        logging.error("the model is fitted. Train your model.")
        raise e
    try:
        assert y_dummy.shape == y_prediction.shape 
    except AssertionError as e:
        logging.error("The predicted shape doesn't match with target shape.")
        raise e