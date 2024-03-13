"""
    This script runs inference on the sliced data.
    Author: Sabari Manohar 
    Date:   March, 2024 
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import logging

from ml.data import process_data # import functions 
from ml.model import train_model, compute_model_metrics # import functions 

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def analyze_feature_slices(data, categorical_features):
    """Evaluates model performance across slices of the dataset based on categorical features.

    Args:
        data (pandas.DataFrame): The dataset containing the features and target variable.
        categorical_features (list): A list of categorical feature names in the dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing evaluation results for each feature slice, 
                          including feature name, category value, precision, recall, and Fbeta score.
    """
    logger.info("Load model")
    model = joblib.load('starter/model/model.pkl')
    encoder = joblib.load('starter/model/encoder.pkl')
    lb = joblib.load('starter/model/lb.pkl')

    logger.info("Split data")
    train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

    results = []  # Store results more efficiently

    for feature in categorical_features:
        for category_value in test_data[feature].unique():
            data_slice = test_data[test_data[feature] == category_value]

            # Only split and process data if the slice is not empty
            if not data_slice.empty:
                X_test, y_test, _, _ = process_data(
                    data_slice, 
                    categorical_features, 
                    label='salary', 
                    training=False,
                    encoder=encoder, 
                    lb=lb
                )

                predictions = model.predict(X_test)
                precision, recall, fbeta = compute_model_metrics(y_test, predictions)

                results.append({
                    'feature': feature,
                    'category': category_value,
                    'precision': precision,
                    'recall': recall,
                    'fbeta': fbeta
                })

    return pd.DataFrame(results)

if __name__ == '__main__':
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
    data = pd.read_csv('starter/data/census_clean.csv')

    slice_results = analyze_feature_slices(data, categorical_features)
    slice_results.to_csv('starter/slice_final_metric.txt', index=False)


