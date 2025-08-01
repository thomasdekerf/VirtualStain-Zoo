# core/metrics_and_image_callback.py
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class MetricsAndImageCallback(Callback):
    def __init__(self, metrics_cfg, image_logging_cfg, data_range=2.0):
        super().__init__()
        self.metrics_cfg = metrics_cfg
        self.image_logging_cfg = image_logging_cfg or {}
        # Metrics
        if self.metrics_cfg.get("psnr", False):
            self.psnr = PeakSignalNoiseRatio(data_range=data_range)
        if self.metrics_cfg.get("ssim", False):
            self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        if self.metrics_cfg.get("fid", False):
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid = FrechetInceptionDistance(feature=64)
        if self.metrics_cfg.get("lpips", False):
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        # Image logging
        self.enable_images = self.image_logging_cfg.get("enable", False)
        self.n_images = self.image_logging_cfg.get("n_images", 4)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        src, tgt = batch
        pred = pl_module(src)
        logs = {}
        if hasattr(self, "psnr"):
            logs["val/psnr"] = self.psnr(pred, tgt)
        if hasattr(self, "ssim"):
            logs["val/ssim"] = self.ssim(pred, tgt)
        if hasattr(self, "fid"):
            self.fid.update((pred + 1) / 2, real=False)
            self.fid.update((tgt + 1) / 2, real=True)
            if batch_idx == 0:
                logs["val/fid"] = self.fid.compute()
                self.fid.reset()
        if hasattr(self, "lpips"):
            logs["val/lpips"] = self.lpips(pred, tgt)
        for k, v in logs.items():
            trainer.logger.log_metrics({k: v.item()}, step=trainer.global_step)
        # Image logging (only on first batch)
        if self.enable_images and batch_idx == 0 and hasattr(trainer.logger, "experiment"):
            n = min(self.n_images, src.size(0))
            grid = torchvision.utils.make_grid(
                torch.cat([src[:n], pred[:n], tgt[:n]], dim=0),
                nrow=n, normalize=True, value_range=(-1, 1)
            )
            # Compatible with TensorBoard, WandB, CSV loggers, etc.
            if hasattr(trainer.logger.experiment, "add_image"):
                trainer.logger.experiment.add_image("val/src_pred_gt", grid, trainer.global_step)
            # Optionally extend with other loggers (e.g., wandb.log)
