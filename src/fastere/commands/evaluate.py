"""CLI command: fastere-evaluate."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import typer

from fastere.config import FastEREConfig

app = typer.Typer(help="Evaluate a trained FastERE model from a checkpoint.")
logger = logging.getLogger(__name__)


@app.command()
def evaluate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to YAML run config (same one used for training).",
    ),
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", "-k", help="Path to model checkpoint (.ckpt file)."
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write evaluation results to JSON."
    ),
    gpu: int = typer.Option(0, "--gpu", help="GPU device index."),
) -> None:
    """Run test-set evaluation on a trained FastERE checkpoint."""

    from fastere.data.data_module import DataModule
    from fastere.models.theta import Theta

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cfg = FastEREConfig.from_yaml(config)
    cfg.training.offline = True
    rc = cfg.to_runtime_config()

    data = DataModule(rc)
    theta = Theta(rc, data)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu],
        precision=rc.precision,
        logger=False,
    )

    trainer.test(theta, datamodule=data, ckpt_path=str(checkpoint))

    result = {
        "test_f1": theta.test_f1,
        "test_f1*": theta.test_f1_plus,
        "test_p": theta.test_p,
        "test_r": theta.test_r,
        "ner_f1": theta.ner_f1,
        "rel_f1": theta.rel_f1,
    }

    typer.echo(json.dumps(result, indent=2))

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        typer.echo(f"\nResults written to {output_file}")

    return result


if __name__ == "__main__":
    app()
