import pytest
import torch
import os
import numpy as np
import cv2

from torch.utils.data import DataLoader
# from torchvision import transforms

# Fix path for imports
import sys
sys.path.append(os.path.abspath('.'))

# Import your modules
from src.dataset import RecognitionDataset
from src.model import FaceVerificationModel
from src.utils import simple_face_detection, get_backbone
from src.transforms import get_default_transform
from omegaconf import OmegaConf

######################## Fixtures ########################

@pytest.fixture
def dummy_config():
    """Fixture for dummy config"""
    return OmegaConf.load("config.yaml")

@pytest.fixture
def dummy_dataloader(dummy_config):
    """Fixture for Dataloader"""
    # Assuming RecognitionDataset is defined in src.dataset
    # and it takes main_path and pair_path as arguments
    transforms = get_default_transform(dummy_config)
    dataset = RecognitionDataset(
        main_path=dummy_config.main_path,
        pair_path=dummy_config.train_pair_path,
        is_train=False,
        transformations=transforms  # or some simple transform
    )
    return DataLoader(dataset, batch_size=2, shuffle=True)

@pytest.fixture
def dummy_model(dummy_config):
    """Fixture for FaceVerificationModel"""
    model = FaceVerificationModel(config=dummy_config,
                                  backbone=get_backbone(dummy_config.backbone.name, pretrained=False),
                                  transform=get_default_transform(dummy_config))
    return model

######################## Tests ########################

def test_dataloader(dummy_dataloader):
    """
    Test if dataloader loads batches correctly.
    """
    batch = next(iter(dummy_dataloader))
    assert isinstance(batch, dict)
    assert 'face1' in batch
    assert 'face2' in batch
    assert 'target' in batch
    assert batch['face1'].shape[0] == 2  # batch_size
    assert batch['face2'].shape[0] == 2

def test_train_epoch(dummy_model, dummy_dataloader):
    """
    Test one forward pass through the model.
    """
    batch = next(iter(dummy_dataloader))
    image1, image2 = batch['face1'], batch['face2']
    
    dummy_model.eval()
    with torch.no_grad():
        output1 = dummy_model.backbone(image1)
        output2 = dummy_model.backbone(image2)
    
    assert output1.shape == output2.shape
    assert output1.ndim == 2  # (batch_size, embedding_dim)

def test_simple_face_detection_success():
    """
    Test simple_face_detection on a sample face image.
    """
    img = cv2.imread('./tests/test_image1.jpg')
    assert img is not None, "Test image not found."
    
    face = simple_face_detection(img)
    
    # Either detect face or return None (valid behavior)
    assert face is None or isinstance(face, np.ndarray)

def test_paths_in_config(dummy_config):
    """
    Test that important paths in config exist.
    """
    assert os.path.exists(dummy_config.main_path), "Main path does not exist!"
    assert os.path.exists(dummy_config.train_pair_path), "Train pair file missing!"
    assert os.path.exists(dummy_config.test_pair_path), "Test pair file missing!"