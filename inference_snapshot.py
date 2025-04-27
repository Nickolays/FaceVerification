import os, hydra, torch
# from src.model import FaceVerificationModel
from src.dataset import RecognitionDataset  # Assuming your dataset class is here
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils import get_backbone
from src.helper import evaluate_model, evaluate_snapshot_ensemble, load_snapshot_models
from src.transforms import get_default_transform

torch.manual_seed(0)


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: DictConfig):
    """
    Evaluate each snapshot model individually and compute the accuracy and ROC AUC.
    Then evaluate the ensemble model by averaging the similarity scores across all snapshot models.
    Print the accuracy and ROC AUC for each snapshot model and the ensemble model.
    The configuration file should contain the following parameters:
    - main_path: The path to the dataset.
    - test_pair_path: The path to the testing pairs.
    - transform: A dictionary containing the parameters for the data transformation.
    - batch_size: The batch size for the dataloader.
    - num_workers: The number of workers for the dataloader.
    - threshold: The threshold for cosine similarity to determine if two faces are similar.

    Work only with cuda
    """
    assert os.path.exists(cfg.main_path)
    assert os.path.exists(cfg.test_pair_path)

    # Prevent Hydra from changing working directory
    os.chdir(HydraConfig.get().runtime.cwd)

    # Data transform
    transform = get_default_transform(cfg)

    # Dataset and dataloader
    test_dataset = RecognitionDataset(cfg.main_path, cfg.test_pair_path, is_train=False, transformations=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Backbone and snapshot models
    backbone = get_backbone(cfg.backbone.name, False)
    snapshot_models = load_snapshot_models("models/snapshots", backbone)
    # snapshot_models = load_snapshot_models("models/snapshots"), cfg, backbone, transform)  # for .ckpt files

    # Evaluate each model individually
    for idx, model in enumerate(snapshot_models):
        acc, roc, _, _, _ = evaluate_model(model, test_loader, threshold=cfg.threshold)
        print(f"Snapshot {idx+1}: Accuracy = {acc:.4f}, ROC AUC = {roc:.4f}" if roc else f"Snapshot {idx+1}: Accuracy = {acc:.4f}")

    # Evaluate ensemble
    ensemble_acc, ensemble_roc = evaluate_snapshot_ensemble(snapshot_models, test_loader, threshold=cfg.threshold)
    print(f"\nEnsemble Performance: Accuracy = {ensemble_acc:.4f}, ROC AUC = {ensemble_roc:.4f}" if ensemble_roc else f"\nEnsemble Performance: Accuracy = {ensemble_acc:.4f}")


if __name__ == "__main__":
    main()  