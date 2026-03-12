import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from core.config import Config
from data.dataloader import MazeDataModule
from models.model import MazeTransformer


def main():
    parser = argparse.ArgumentParser(description="Train a maze transformer")
    parser.add_argument(
        "--config", type=str, default="config/base.toml", help="Path to the config file"
    )
    args = parser.parse_args()

    config = Config.load(args.config)

    L.seed_everything(config.input_data.seed)

    datamodule = MazeDataModule(
        npz_path="data/maze_data.npz", batch_size=config.input_data.batch_size
    )

    model = MazeTransformer(config=config)

    logger = WandbLogger(
        project=config.logging.project_name,
        name=config.project.experiment_name,
        save_dir="experiments",
    )

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints",
        filename="maze-{step}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        precision="16-mixed",
        gradient_clip_val=config.training.gradient_clipping,
        accelerator="auto",
        logger=logger,
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        val_check_interval=config.training.eval_interval,
        log_every_n_steps=config.training.log_interval,
    )

    trainer.fit(model=model, datamodule=datamodule)

    if __name__ == "__main__":
        main()
