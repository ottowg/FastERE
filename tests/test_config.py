"""Tests for FastEREConfig parsing and RuntimeConfig conversion."""

from pathlib import Path

import pytest
import yaml

from fastere.config import (
    DatasetConfig,
    EnvSettings,
    FastEREConfig,
    ModelConfig,
    TrainingConfig,
    _SimpleConfig,
)


class TestDatasetConfig:
    def test_basic_construction(self, minimal_dataset_config):
        cfg = DatasetConfig(**minimal_dataset_config)
        assert cfg.name == "test_dataset"
        assert "PER" in cfg.ents
        assert "PART-OF" in cfg.rels
        assert cfg.max_seq_len == 512

    def test_get_method(self, minimal_dataset_config):
        cfg = DatasetConfig(**minimal_dataset_config)
        assert cfg.get("name") == "test_dataset"
        assert cfg.get("nonexistent", "default") == "default"

    def test_getitem(self, minimal_dataset_config):
        cfg = DatasetConfig(**minimal_dataset_config)
        assert cfg["ents"] == cfg.ents


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(name="BERT", model_name_or_path="bert-base-uncased")
        assert cfg.name == "BERT"
        assert cfg.hidden_size is None

    def test_extra_fields_allowed(self):
        cfg = ModelConfig(name="X", model_name_or_path="y", custom_field=42)
        assert cfg.custom_field == 42


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.seed == 42
        assert cfg.lr == 3e-5
        assert cfg.precision == 16
        assert cfg.use_filter is True

    def test_get(self):
        cfg = TrainingConfig(tag="my-run")
        assert cfg.get("tag") == "my-run"
        assert cfg.get("nonexistent", 99) == 99


class TestFastEREConfig:
    def test_from_yaml(self, minimal_config_yaml):
        cfg = FastEREConfig.from_yaml(minimal_config_yaml)
        assert cfg.model.name == "BERT"
        assert cfg.dataset.name == "test_dataset"
        assert cfg.training.tag == "test-run"
        assert cfg.training.max_epochs == 3

    def test_env_defaults(self, minimal_config_yaml, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # no .env file here — pydantic defaults apply
        monkeypatch.delenv("DATASET_DIR", raising=False)
        monkeypatch.delenv("OUTPUT_DIR", raising=False)
        cfg = FastEREConfig.from_yaml(minimal_config_yaml)
        assert cfg.env.dataset_dir == "datasets"
        assert cfg.env.output_dir == "output"

    def test_flat_training_keys(self, minimal_config_dict, tmp_path):
        # Training keys can also be at top level (convenience)
        cfg_dict = {**minimal_config_dict, "seed": 99}
        config_path = tmp_path / "config2.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f)
        cfg = FastEREConfig.from_yaml(config_path)
        assert cfg.training.seed == 99


class TestSimpleConfig:
    def test_case_insensitive(self):
        sc = _SimpleConfig({"Hello": "world"})
        assert sc["Hello"] == "world"
        assert sc["hello"] == "world"
        assert sc.hello == "world"

    def test_setattr(self):
        sc = _SimpleConfig()
        sc.MyKey = "value"
        assert sc["mykey"] == "value"
        assert sc.mykey == "value"
