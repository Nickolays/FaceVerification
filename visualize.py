import os
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch.nn.functional as F

from src.utils import get_backbone, min_max_scaler


# Load the config.yaml file as a DictConfig object
cfg = OmegaConf.load("config.yaml")


def visualize_pairs(cfg: DictConfig):
    """
    Visualizes N face verification pairs from a CSV with similarity score and target label.
    Uses the loaded model to compute cosine similarity on transformed image pairs.
    """

    assert os.path.exists(cfg.infer.image_root)
    assert os.path.exists(cfg.infer.csv_path)
    assert os.path.exists(cfg.infer.model_path)
    # Load model
    transform = transforms.Compose([
        transforms.Resize((cfg.transform.resize, cfg.transform.resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    model = get_backbone(cfg.backbone.name, pretrained=False) # replace with actual backbone
    # model.load_state_dict(torch.load(cfg.infer.model_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(cfg.infer.model_path, 
                                map_location='cpu', 
                                weights_only=False))
    model.eval()

    # Load CSV with image pairs
    df = pd.read_csv(cfg.infer.csv_path).sample(cfg.infer.num_pairs)
    images = []
    labels = []

    for _, row in df.iterrows():
        path1 = os.path.join(cfg.infer.image_root, row["face1"])
        path2 = os.path.join(cfg.infer.image_root, row["face2"])
        target = row["target"]

        # Load & transform
        img1 = transform(Image.open(path1).convert("RGB")).unsqueeze(0)
        img2 = transform(Image.open(path2).convert("RGB")).unsqueeze(0)

        # Compute cosine similarity
        with torch.no_grad():
            emb1 = model(img1)
            emb2 = model(img2)
            sim = F.cosine_similarity(emb1, emb2).item()

        img1 = min_max_scaler(img1.squeeze())
        img2 = min_max_scaler(img2.squeeze())
        images.extend([img1, img2])
        labels.append(f"Target: {target} | Sim: {sim:.2f} |  Face1: {row['face1']} | Face2: {row['face2']}")

    # Visualize grid
    grid = utils.make_grid(images, nrow=2, padding=8, pad_value=1)

    plt.figure(figsize=(6, 3 * cfg.infer.num_pairs))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")

    for i, text in enumerate(labels):
        y = i * (cfg.transform.resize + 8) + cfg.transform.resize // 2
        plt.text(0, y, text, fontsize=10, color='white',
                 bbox=dict(facecolor='black', alpha=0.7))
    plt.title("Face Verification Pairs", fontsize=16)   
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs("results", exist_ok=True)
    # Save the figure
    plt.savefig("results/visualized_grid.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    visualize_pairs(cfg)