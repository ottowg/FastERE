"""Shared pytest fixtures."""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def minimal_dataset_config() -> dict:
    return {
        "name": "test_dataset",
        "ents": ["PER", "ORG", "LOC"],
        "rels": ["PART-OF", "WORKS-FOR"],
        "train": "data/train.json",
        "val": "data/dev.json",
        "test": "data/test.json",
    }


@pytest.fixture
def minimal_config_dict(minimal_dataset_config, tmp_path) -> dict:
    return {
        "model": {"name": "BERT", "model_name_or_path": "bert-base-uncased"},
        "dataset": minimal_dataset_config,
        "training": {"tag": "test-run", "max_epochs": 3, "batch_size": 2},
    }


@pytest.fixture
def minimal_config_yaml(minimal_config_dict, tmp_path) -> Path:
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(minimal_config_dict, f)
    return config_path
