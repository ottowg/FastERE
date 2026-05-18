"""CLI command: fastere-train."""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import typer
import wandb

from fastere.config import FastEREConfig

app = typer.Typer(help="Train a FastERE model.")
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML run config."),
    tag: Optional[str] = typer.Option(
        None, "--tag", "-t", help="Override experiment tag."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug mode (fast, offline)."
    ),
    offline: bool = typer.Option(False, "--offline", help="Run W&B in offline mode."),
    max_epochs: Optional[int] = typer.Option(
        None, "--max-epochs", help="Override max training epochs."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override batch size."
    ),
    lr: Optional[float] = typer.Option(None, "--lr", help="Override learning rate."),
    precision: Optional[int] = typer.Option(
        None, "--precision", help="Training precision (16 or 32)."
    ),
    gpu: int = typer.Option(0, "--gpu", help="GPU device index."),
) -> None:
    """Train a FastERE entity-relation extraction model from a YAML config."""

    # Lazy imports to keep startup fast
    from fastere.data.data_module import DataModule
    from fastere.models.theta import Theta

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cfg = FastEREConfig.from_yaml(config)

    # CLI overrides
    if tag:
        cfg.training.tag = tag
    if debug:
        cfg.training.debug = True
        cfg.training.offline = True
    if offline:
        cfg.training.offline = True
    if max_epochs is not None:
        cfg.training.max_epochs = max_epochs
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    if lr is not None:
        cfg.training.lr = lr
    if precision is not None:
        cfg.training.precision = precision

    rc = cfg.to_runtime_config()
    seed_everything(rc.seed)

    data = DataModule(rc)
    theta = Theta(rc, data)

    early_callback = pl.callbacks.EarlyStopping(
        monitor="val_f1", mode="max", patience=30, check_on_train_epoch_end=False
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_last=False,
        save_top_k=1,
        filename="f1={val_f1:.4f}-epoch={epoch}",
        dirpath=os.path.join(rc.output_dir, "checkpoints"),
        auto_insert_metric_name=False,
        save_weights_only=True,
    )

    wandb_logger = _configure_logger(rc, cfg)

    trainer = pl.Trainer(
        max_epochs=rc.max_epochs,
        accelerator="gpu",
        devices=[gpu],
        precision="16-mixed" if rc.precision == 16 else rc.precision,
        callbacks=[early_callback, model_checkpoint],
        log_every_n_steps=10,
        default_root_dir=rc.output_dir,
        logger=wandb_logger,
        fast_dev_run=rc.fast_dev_run,
    )

    if rc.auto_lr:
        from pytorch_lightning.tuner import Tuner

        typer.echo("Finding optimal learning rate...")
        Tuner(trainer).lr_find(theta, datamodule=data)

    trainer.fit(theta, datamodule=data)
    rc.save_best_model_path(model_checkpoint.best_model_path)
    rc.save_config()

    if model_checkpoint.best_model_path:
        trainer.test(theta, datamodule=data, ckpt_path=model_checkpoint.best_model_path)

    wandb.finish(quiet=True)

    result = {
        "best_f1": theta.best_f1,
        "test_f1": theta.test_f1,
        "test_f1*": theta.test_f1_plus,
        "test_p": theta.test_p,
        "test_r": theta.test_r,
        "ner_f1": theta.ner_f1,
        "rel_f1": theta.rel_f1,
    }
    rc.save_result(result)
    typer.echo(f"\nTraining complete. Results: {result}")
    return result


def _configure_logger(rc, cfg: FastEREConfig):
    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)

    return pl.loggers.WandbLogger(
        project=cfg.env.wandb_project,
        name=rc.tag,
        save_dir=rc.output_dir,
        offline=rc.offline,
        save_code=True,
    )


if __name__ == "__main__":
    app()
