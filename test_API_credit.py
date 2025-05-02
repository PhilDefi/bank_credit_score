from fastapi.testclient import TestClient
from API_credit import app  # Assure-toi que le nom du fichier est API_credit.py

client = TestClient(app)

# Test predict/shape endpoit : unique row shape
def test_data_shape():
    payload = {
        "data": [["1", "2", "3"]],
        "columns": ["col1", "col2", "col3"]
    }

    response = client.post("/data_shape", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {
        "message": "Received your request!",
        "data_shape": [1, 3]
    }

# Test predict/shape endpoit : multiple rows shape
def test_data_shape_multiple_rows():
    payload = {
        "data": [["1", "2", "3"], ["4", "5", "6"]],
        "columns": ["col1", "col2", "col3"]
    }

    response = client.post("/data_shape", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {
        "message": "Received your request!",
        "data_shape": [2, 3]
    }