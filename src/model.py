import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score

from torch.utils.data import DataLoader

from .dataset import RecognitionDataset
from .losses import PairwiseTripletLoss, ArcFaceLoss


class FaceVerificationModel(pl.LightningModule):
    def __init__(self, config, backbone, transform):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "transform"])

        self.config = config
        self.backbone = backbone
        self.transform = transform

        self.lr = config.lr
        self.threshold = config.threshold
        self.loss_type = config.loss_type
        self.main_path = config.main_path
        self.train_pair_path = config.train_pair_path
        self.test_pair_path = config.test_pair_path
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        if hasattr(self.config, "finetuning") and self.config.finetuning.freeze_n_layers > 0:
            self.freeze_backbone_layers(self.config.finetuning.freeze_n_layers)

        if self.loss_type == "triplet":
            self.loss_fn = PairwiseTripletLoss(margin=0.5)
        elif self.loss_type == "arcface":
            self.loss_fn = ArcFaceLoss(margin=config.margin, scale=config.scale)
        else:
            raise ValueError("Unsupported loss type")

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x1, x2, target = batch['face1'], batch['face2'], batch['target']
        emb1, emb2 = self(x1), self(x2)
        loss = self.loss_fn(emb1, emb2, target)
        self.log('train_loss', loss, prog_bar=True)
        # TensorBoard logging
        if hasattr(self, 'writer'):
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, 'validation')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, 'test')

    def _shared_eval_step(self, batch, stage):
        x1, x2, target = batch['face1'], batch['face2'], batch['target']
        emb1, emb2 = self(x1), self(x2)
        similarity = F.cosine_similarity(emb1, emb2)
        pred = (similarity > self.threshold).long()
        loss = self.loss_fn(emb1, emb2, target)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        output = {'preds': pred.detach().cpu(), 'targets': target.detach().cpu(), 'scores': similarity.detach().cpu()}
        getattr(self, f'{stage}_step_outputs').append(output)
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f"{stage}/loss", loss.item(), self.global_step)
        return loss

    def on_train_epoch_end(self):
        self.train_dataset.on_epoch_end()
        if hasattr(self, 'writer') and self.trainer is not None:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.writer.add_scalar('epoch/learning_rate', lr, self.current_epoch)

    def on_validation_epoch_end(self):
        self._compute_and_log_epoch_metrics(self.validation_step_outputs, 'validation')
        self.validation_step_outputs.clear()
        self.valid_dataset.on_epoch_end()

    def on_test_epoch_end(self):
        self._compute_and_log_epoch_metrics(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def _compute_and_log_epoch_metrics(self, outputs, stage):
        preds = torch.cat([o['preds'] for o in outputs])
        targets = torch.cat([o['targets'] for o in outputs])
        scores = torch.cat([o['scores'] for o in outputs])
        acc = accuracy_score(targets, preds)
        try:
            roc = roc_auc_score(targets, scores)
        except:
            roc = None
        self.log(f"{stage}_epoch_acc", acc, prog_bar=True)
        if roc is not None:
            self.log(f"{stage}_epoch_roc_auc", roc, prog_bar=True)
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f"{stage}/acc", acc, self.global_step)
            if roc is not None:
                self.writer.add_scalar(f"{stage}/roc_auc", roc, self.global_step)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)

    def setup(self, stage=None):
        self.train_dataset = RecognitionDataset(self.main_path, self.train_pair_path, is_train=True, transformations=self.transform)
        self.valid_dataset = RecognitionDataset(self.main_path, self.test_pair_path, is_train=False, transformations=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def freeze_backbone_layers(self, n_layers: int):
        count = 0
        for child in self.backbone.children():
            for param in child.parameters():
                param.requires_grad = False
            count += 1
            if count >= n_layers:
                break

