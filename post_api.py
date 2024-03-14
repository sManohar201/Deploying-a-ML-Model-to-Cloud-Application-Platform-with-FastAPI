"""
    This script is to post to cloud platform for inference
    Author: Sabari Manohar 
    Date:   March, 2024 
"""

import requests
import json

url = "https://cloud-application-with-fastapi.onrender.com/predict"


# explicit the sample to perform inference on
data =  { 'age':50,
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

# post to cloud API get response
response = requests.post(url, json=data)

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())
