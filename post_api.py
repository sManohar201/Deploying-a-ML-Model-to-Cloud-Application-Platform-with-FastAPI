"""
    This script is to post to cloud platform for inference
    Author: Sabari Manohar 
    Date:   March, 2024 
"""

import requests
import json

url = "https://cloud-application-with-fastapi.onrender.com/predict"


# explicit the sample to perform inference on
data = {"age": 47,
            "workclass": "Private",
            "fnlgt": 51835,
            "education": "Prof-school",
            "education_num": 15,
            "marital_status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 1902,
            "hours_per_week": 60,
            "native_country": "Honduras"
            }

# post to cloud API get response
response = requests.post(url, json=data)

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())
