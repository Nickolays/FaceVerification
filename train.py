import hydra, os, torch
from omegaconf import DictConfig
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.callbacks import SnapshotEnsembling    
from src.model import FaceVerificationModel
from src.utils import get_backbone, create_writer  # Replace with your model import


# Set up checkpoint callback to save best model based on val accuracy
checkpoint_callback = ModelCheckpoint(
    monitor="validation_epoch_acc",       # your custom validation metric
    mode="max",                    # maximize accuracy
    save_top_k=1,
    dirpath="models",
    filename="face-verification-{epoch:02d}-{validation_epoch_acc:.4f}",
    save_weights_only=False        # saves full model (weights + config)
)
# Set up snapshot ensembling callback   
snapshot_callback = SnapshotEnsembling(
    start_learning_rate=1e-3,
    end_learning_rate=1e-5,
    n_epochs_to_restart=5,
    n_snapshots=3,
    save_path="models/snapshots"
)


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to train the FaceVerificationModel using PyTorch Lightning.
    Expects a configuration file named "config.yaml" in the current working directory.
    The configuration file should contain the following parameters:
    - epochs: The number of epochs to train the model.
    - backbone: The name of the backbone model.
    - main_path: The path to the dataset.
    - train_pair_path: The path to the training pairs.
    - test_pair_path: The path to the testing pairs.
    - transform: A dictionary containing the parameters for the data transformation.
    - transform.resize: The size of the resized images.
    - backbone.pretrained: Whether to use a pre-trained backbone model.
    """
    assert os.path.exists(cfg.main_path)
    assert os.path.exists(cfg.train_pair_path) and os.path.exists(cfg.test_pair_path)

    # Prevent Hydra from changing working directory
    from hydra.core.hydra_config import HydraConfig
    os.chdir(HydraConfig.get().runtime.cwd)

    transform = transforms.Compose([
        transforms.Resize((cfg.transform.resize, cfg.transform.resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 
    ])

    # Initialize TensorBoard writer
    writer = create_writer(log_dir="tensorboard_logs", model_name=cfg.backbone.name)

    # Initialize the backbone model    
    backbone = get_backbone(cfg.backbone.name, pretrained=cfg.backbone.pretrained) # replace with actual backbone
    model = FaceVerificationModel(config=cfg, backbone=backbone, transform=transform)
    model.writer = writer  # <- Optional: pass writer to model for logging custom metrics

    trainer = Trainer(
        max_epochs=cfg.epochs, 
        accelerator="auto",
        precision=16,
        # limit_train_batches=0.1,  # Use 10% of the training data
        # max_steps=10000,
        accumulate_grad_batches=5,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, snapshot_callback],
        logger=False  # Disable Lightning’s default logger if you're using SummaryWriter manually
    )
    trainer.fit(model)

    # Optionally save final model manually (after training)
    save_path = os.path.join("models", "final_model.ckpt")
    try:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # The best checkpoint will be saved in trainer.checkpoint_callback.best_model_path
    best_model = FaceVerificationModel.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        config=cfg,
        backbone=backbone,
        transform=transform
    )

    torch.save(best_model.backbone.state_dict(), "models/face_verification_backbone.pth")
    torch.save(best_model.state_dict(), "models/face_verification_model.cpkt")

if __name__ == "__main__":
    main()  