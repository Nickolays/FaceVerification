import os
import math
import torch
from pytorch_lightning.callbacks import Callback


class SnapshotEnsembling(Callback):
    def __init__(
        self,
        start_learning_rate: float,
        end_learning_rate: float,
        n_epochs_to_restart: int,
        n_snapshots: int,
        save_path: str = "snapshots"
    ):
        super().__init__()
        self.start_lr = start_learning_rate
        self.end_lr = end_learning_rate
        self.T = n_epochs_to_restart  # number of iterations before restart
        self.M = n_snapshots         # number of snapshots to save
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def on_train_start(self, trainer, pl_module):
        self.total_epochs = trainer.max_epochs
        self.epochs_per_cycle = self.total_epochs // self.M
        self.snapshot_idx = 0

    def on_train_epoch_start(self, trainer, pl_module):
        # Cosine annealing
        epoch = trainer.current_epoch
        cycle_epoch = epoch % self.epochs_per_cycle
        lr = self.end_lr + 0.5 * (self.start_lr - self.end_lr) * (
            1 + math.cos(math.pi * cycle_epoch / self.epochs_per_cycle)
        )
        for param_group in pl_module.trainer.optimizers[0].param_groups:
            param_group["lr"] = lr

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.epochs_per_cycle == 0 and self.snapshot_idx < self.M:
            # Get the backbone model from the PyTorch Lightning module
            backbone_model = pl_module.backbone  # assuming the backbone model is an attribute of the PL module

            # Save the backbone model as a PyTorch model
            filename = os.path.join(self.save_path, f"snapshot_{self.snapshot_idx}.pth")
            torch.save(backbone_model.state_dict(), filename)
            print(f"Saved snapshot {self.snapshot_idx} at epoch {epoch + 1} to {filename}")
            self.snapshot_idx += 1