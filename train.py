import hydra, pytorch_lightning as pl
from omegaconf import DictConfig
from core.registry import get_model
import datasets.pair_folder_datamodule as dm_mod
from core.metrics_and_image_callback import MetricsAndImageCallback

# train.py  (top of file)
# Default config is passed at runtime via --config-name
@hydra.main(config_path='configs', config_name=None)
def main(cfg):
    print("Experiment:", cfg.experiment.name)
    print("Description:", cfg.experiment.description)

    # pass the metrics dict to your model
    model = get_model(**cfg.model, metrics_cfg=cfg.get("metrics", {}))

    # Toggle logger based on config
    loggers = []
    if cfg.logger.get("tensorboard", False):
        from pytorch_lightning.loggers import TensorBoardLogger
        # loggersTensorBoardLogger("lightning_logs/"))
        exp_name = cfg.experiment.get("name", "unnamed")
        loggers.append(TensorBoardLogger(save_dir="all_tb_logs", name=exp_name))

    if cfg.logger.get("wandb", False):
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(project="myproject"))

    callbacks = [
        MetricsAndImageCallback(
            metrics_cfg=cfg.get("metrics", {}),
            image_logging_cfg=cfg.get("image_logging", {}),
            data_range=2.0
        )
    ]
    trainer = pl.Trainer(logger=loggers if loggers else None,
                         callbacks=callbacks, **cfg.trainer)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    import os

    print("CWD at start of training:", os.getcwd())

    main()
