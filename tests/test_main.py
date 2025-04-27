import pytest
import os
import sys

from fastapi.testclient import TestClient
from PIL import Image

# Fix import paths
dynamic_path = os.path.abspath('.')
sys.path.append(dynamic_path)

# Import your FastAPI app
from main import app

############################ Fixtures ############################

@pytest.fixture
def test_images():
    """Fixture to load two test images"""
    files = {
        'image1': open('./tests/test_image1.jpg', 'rb'),
        'image2': open('./tests/test_image2.jpg', 'rb')
    }
    return files

@pytest.fixture
def test_client():
    """Fixture to create FastAPI test client"""
    return TestClient(app)

############################ Tests ############################

def test_healthcheck(test_client):
    """
    Test the /healthcheck endpoint to verify app is running.
    """
    response = test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"healthcheck": "Everything OK!"}

def test_verify_images_success(test_client, test_images):
    """
    Test the /verify_images endpoint with valid face images.
    """
    response = test_client.post("/verify_images", files=test_images)
    
    # Check response status
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    # Check response structure
    data = response.json()
    assert "similarity" in data
    assert isinstance(data["similarity"], float)

def test_verify_images_failure(test_client):
    """
    Test /verify_images endpoint with missing images.
    Should return 422 (Unprocessable Entity).
    """
    response = test_client.post("/verify_images", files={})
    assert response.status_code == 422  # FastAPI validation error
