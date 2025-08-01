
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch

class BaseVSModel(pl.LightningModule, ABC):
    """Abstract base class for all VirtualStain models."""
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    @abstractmethod
    def forward(self, x):
        """Return virtual‐stained output given source image tensor x."""
        pass

    def loss_fn(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        pred = self(src)
        loss = self.loss_fn(pred, tgt)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        pred = self(src)
        loss = self.loss_fn(pred, tgt)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get('lr', 2e-4), betas=(0.5, 0.999))
