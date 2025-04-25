import argparse
import logging
import os

# from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision import transforms

from src.utils import get_backbone, simple_face_detection  # Replace with your actual imports


def setup_logger():
    logger = logging.getLogger("FaceVerification")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main(config_path, img_path1, img_path2, output_dir):
    """
    Main entry point for the script.

    Args:
        config_path (str): Path to the configuration file.
        img_path1 (str): Path to the first image.
        img_path2 (str): Path to the second image.
        output_dir (str): Directory to save the preprocessed images and log the cosine similarity.

    Returns:
        None
    """
    logger = setup_logger()

    # Load config
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.transform.resize, cfg.transform.resize)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load model backbone and weights
    model = get_backbone(cfg.backbone.name, pretrained=False)
    logger.info(f"Loading model weights from {cfg.infer.model_path}")
    model.load_state_dict(torch.load(cfg.infer.model_path, map_location='cpu', weights_only=False))
    model.eval()

    # Read and preprocess images
    assert os.path.exists(img_path1), f"Image not found: {img_path1}"
    assert os.path.exists(img_path2), f"Image not found: {img_path2}"
    logger.info(f"Reading images: {img_path1}, {img_path2}")
    img1 = simple_face_detection(cv2.imread(img_path1))
    img2 = simple_face_detection(cv2.imread(img_path2))

    os.makedirs(output_dir, exist_ok=True)
    plt.imsave(os.path.join(output_dir, 'img1.jpg'), img1)
    plt.imsave(os.path.join(output_dir, 'img2.jpg'), img2)
    logger.info(f"Saved preprocessed images to {output_dir}")

    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output1 = model(img1)
        output2 = model(img2)

    similarity = torch.nn.functional.cosine_similarity(output1, output2)
    logger.info(f"Cosine Similarity: {similarity.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Verification Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--img1", type=str, required=True, default='tests/Adam_Sandler_0001.jpg', help="Path to first test image")
    parser.add_argument("--img2", type=str, required=True, default='tests/Adam_Sandler_0003.jpg', help="Path to second test image")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output images")

    args = parser.parse_args()

    main(args.config, args.img1, args.img2, args.output_dir)

# WITHOUT CLI
# from PIL import Image
# import hydra, os, torch
# from omegaconf import DictConfig, OmegaConf
# import cv2
# import matplotlib.pyplot as plt

# from torchvision import transforms

# from src.utils import get_backbone, simple_face_detection  # Replace with your model import


# # Load the config.yaml file as a DictConfig object
# cfg = OmegaConf.load("config.yaml")

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((cfg.transform.resize, cfg.transform.resize)),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 
# ])

# # load backbone and load weigths on cpu
# model = get_backbone(cfg.backbone.name, pretrained=False) # replace with actual backbone
# # model.load_state_dict(torch.load(cfg.infer.model_path, map_location=torch.device('cpu')))
# model.load_state_dict(torch.load(cfg.infer.model_path, 
#                             map_location='cpu', 
#                             weights_only=False))
# model.eval()

# # Test images
# path_1 = 'tests/Adam_Sandler_0001.jpg'
# path_2 = 'tests/Adam_Sandler_0003.jpg'

# img1 = simple_face_detection(cv2.imread(path_1))
# img2 = simple_face_detection(cv2.imread(path_2))

# plt.imsave('results/img1.jpg', img1)
# plt.imsave('results/img2.jpg', img2)

# img1 = transform(img1)
# img2 = transform(img2)
# # Add batch dimension
# img1 = img1.unsqueeze(0)
# img2 = img2.unsqueeze(0)

# # Inference model
# output1 = model(img1)
# output2 = model(img2)

# # Calculate cosine similarity
# similarity = torch.nn.functional.cosine_similarity(output1, output2)
# print(f"Cosine Similarity: {similarity.item()}")