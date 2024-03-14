"""
    This script contains pytest modules for the model.
    Author: Sabari Manohar 
    Date: March, 2024 
"""

import sys
sys.path.append('.')  # Ensure project modules are in search path
import joblib
import pytest
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data  # Adjusted module name
from ml.model import inference, train_model

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
    loaded_model = joblib.load('model/model.pkl')  
    assert isinstance(loaded_model, RandomForestClassifier)

def test_data_integrity():
    """Checks if the dataset can be loaded and contains at least one data point.
    """
    data_path = 'data/census_clean.csv'
    df = pd.read_csv(data_path)
    assert df.shape[0] > 0  # Ensure there are rows

def test_processing_consistency():
    """Ensures that data processing produces feature and target arrays with compatible shapes.
    """
    data_path = 'data/census_clean.csv'
    df = pd.read_csv(data_path)
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42) 

    X_train, y_train, _, _ = process_data(
        df_train, categorical_features, label="salary", training=True
    )
    assert X_train.shape[0] == y_train.shape[0]

def test_inference():
    """Ensures the training pipeline runs as expected and inference is right.
    """
    x_dummy = np.random.rand(30, 5)
    y_dummy = np.random.randint(2, size=30)

    model_dummy = train_model(x_dummy, y_dummy)
    y_prediction = inference(model_dummy, x_dummy)

    assert y_dummy.shape == y_prediction.shape 