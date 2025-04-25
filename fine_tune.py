import hydra, os, torch
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model import FaceVerificationModel
from src.utils import get_backbone, create_writer
from src.callbacks import SnapshotEnsembling  # if you want snapshot in fine-tune


# === Fine-tune Parameters ===
PRETRAINED_PATH = "models/face-verification-epoch=14-validation_epoch_acc=0.9957.ckpt"  # path to pretrained model
FINE_TUNE_DIR = "models/fine_tuned"
os.makedirs(FINE_TUNE_DIR, exist_ok=True)

# === Checkpoint for fine-tuning ===
checkpoint_callback = ModelCheckpoint(
    monitor="validation_epoch_acc",
    mode="max",
    save_top_k=1,
    dirpath=FINE_TUNE_DIR,
    filename="fine-tuned-{epoch:02d}-{validation_epoch_acc:.4f}",
)

@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: DictConfig):
    # Prevent Hydra from changing working directory
    from hydra.core.hydra_config import HydraConfig
    os.chdir(HydraConfig.get().runtime.cwd)

    # # === Modify config paths if needed ===
    # cfg.main_path = "/path/to/your/new_dataset"
    # cfg.train_pair_path = "/path/to/your/train_pairs.csv"
    # cfg.test_pair_path = "/path/to/your/test_pairs.csv"

    # === Setup Transform ===
    transform = transforms.Compose([
        transforms.Resize((cfg.transform.resize, cfg.transform.resize)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.transform.normalize_mean,
            std=cfg.transform.normalize_std
        )
    ])

    # === Load Backbone and weights ===
    backbone = get_backbone(cfg.backbone.name, pretrained=False)
    # backbone.load_state_dict(torch.load(PRETRAINED_BACKBONE_PATH, map_location='cpu'))

    # === Optional: freeze some backbone layers ===
    # for param in backbone.parameters():
    #     param.requires_grad = False
    # for param in backbone.layer4.parameters():
    #     param.requires_grad = True  # fine-tune only last block

    # === Load model checkpoint and fine-tune ===
    model = FaceVerificationModel.load_from_checkpoint(
        checkpoint_path=PRETRAINED_PATH,
        config=cfg,
        backbone=backbone,
        transform=transform
    )
    model.writer = create_writer(log_dir="tensorboard_logs", model_name=cfg.backbone.name)

    # Re-train on your dataset
    trainer = Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        precision=16,
        accumulate_grad_batches=5,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],  # optionally add snapshot again
        logger=False
    )
    trainer.fit(model)

    print("Fine-tuning complete.")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Save final backbone state_dict
    torch.save(model.backbone.state_dict(), os.path.join(FINE_TUNE_DIR, "finetuned_backbone.pth"))

if __name__ == "__main__":
    main()
