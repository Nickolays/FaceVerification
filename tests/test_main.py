import pytest
import os
import sys
import torch
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
    files = {
        'image1': open('tests/test_image1.jpg', 'rb').read(),
        'image2': open('tests/test_image2.jpg', 'rb').read()
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

def test_load_images():
    """
    Test loading images from the fixture.
    """
    assert os.path.exists('tests/test_image1.jpg')
    assert os.path.exists('tests/test_image2.jpg')

def test_preprocess_image(test_images):
    """
    Test the image preprocessing function.
    """
    from main import preprocess
    image1 = test_images['image1']
    image2 = test_images['image2']

    # Preprocess images
    processed_image1 = preprocess(image1)
    processed_image2 = preprocess(image2)

    # Check if the output is a tensor
    assert isinstance(processed_image1, torch.Tensor)
    assert isinstance(processed_image2, torch.Tensor)

def test_verify_images_failure(test_client):
    """
    Test /verify_images endpoint with missing images.
    Should return 422 (Unprocessable Entity).
    """
    response = test_client.post("/verify_images", files={})
    assert response.status_code == 422  # FastAPI validation error

# def test_verify_images_success(test_client, test_images):

#     response = test_client.post("/verify_images", files=test_images)

#     assert response.status_code == 200
#     assert response.headers["content-type"] == "application/json"

#     data = response.json()
#     assert "similarity" in data
#     assert isinstance(data["similarity"], float)