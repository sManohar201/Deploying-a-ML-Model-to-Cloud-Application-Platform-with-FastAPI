from fastapi.testclient import TestClient
import os
import sys
import json

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from main import api

client = TestClient(api)


def test_root():
    """
        welcome message at root test
    """
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == 'Welcome to the Income Prediction Service!'


def test_inference_zero():
    """
    Test model inference output
    """
    data = {"age": 52,
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

    res = client.post("/predict", json=data )

    # test output response
    assert res.status_code == 200
    assert res.json()["age"] == 52
    assert res.json()["fnlgt"] == 287927
    assert res.json()["prediction"] == '>50K'

def test_inference_one():
    """
    Test model inference output 
    """
    data =  {  'age':30,
                'workclass':"Private", 
                'fnlgt':234721,
                'education':"HS-grad",
                'education_num':1,
                'marital_status':"Separated",
                'occupation':"Handlers-cleaners",
                'relationship':"Not-in-family",
                'race':"Black",
                'sex':"Male",
                'capital_gain':0,
                'capital_loss':0,
                'hours_per_week':35,
                'native_country':"United-States"
            }
    # data = json.dumps(data)

    res = client.post("/predict", json=data )

    # test output response
    assert res.status_code == 200
    assert res.json()["age"] == 30
    assert res.json()["fnlgt"] == 234721
    assert res.json()["prediction"] == '<=50K'


def test_predict_invalid():
    data = {}
    response = client.post("/predict", json=json.dumps(data))
    assert response.status_code == 422


if __name__ == "__main__":
    test_root()
    test_inference_one()
    test_inference_zero()
    test_predict_invalid()