"""Pydantic-based configuration for FastERE.

Load a full run config from a YAML file:

    cfg = FastEREConfig.from_yaml("configs/my_run.yaml")

The YAML structure mirrors the model fields below.  Path values (dataset_dir,
output_dir) can also be supplied via the .env file — see EnvSettings.
"""

from __future__ import annotations

import json
import os
import re
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Environment / path settings (read from .env)
# ---------------------------------------------------------------------------


class EnvSettings(BaseSettings):
    """Project-wide paths and environment settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dataset_dir: str = "datasets"
    output_dir: str = "output"
    cache_dir: str = ".cache"
    wandb_project: str = "fastere"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    name: str = "BERT"
    model_name_or_path: str = "bert-base-uncased"

    # Fields populated at runtime from the HuggingFace config
    hidden_size: Optional[int] = None
    hidden_dropout_prob: Optional[float] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    model_type: Optional[str] = None

    model_config = {"extra": "allow"}

    def load_hf_config(self) -> None:
        """Populate hidden_size etc. from the pretrained HuggingFace config."""
        hf_cfg = AutoConfig.from_pretrained(self.model_name_or_path)
        for k, v in hf_cfg.to_dict().items():
            if not hasattr(self, k) or getattr(self, k) is None:
                object.__setattr__(self, k, v)


class DatasetConfig(BaseModel):
    name: str
    rels: List[str]
    ents: List[str]
    na_label: Optional[str] = None
    max_seq_len: int = 512
    train: Union[str, List[str]] = ""
    val: Union[str, List[str]] = ""
    test: Union[str, List[str]] = ""

    model_config = {"extra": "allow"}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class TrainingConfig(BaseModel):
    seed: int = 42
    lr: float = 3e-5
    batch_size: int = 4
    test_batch_size: int = 0
    max_epochs: int = 100
    precision: int = 16
    num_worker: int = 8
    tag: str = "experiment"
    debug: bool = False
    offline: bool = False
    wandb: bool = False
    auto_lr: bool = False
    use_cache: bool = True
    global_batch_size: Optional[int] = None

    # Model component rates / warmup
    ner_rate: float = 1.0
    rel_rate: float = 1.0
    filter_rate: float = 1.0
    rel_ner_rate: float = 0.0
    ner_lr: float = 1.0
    rel_lr: float = 1.0
    filter_lr: float = 1.0
    warmup: float = 0.06
    optimizer: str = "AdamW"

    # Feature flags (all False by default — enable in your YAML as needed)
    use_filter: bool = True
    use_ner_prompt: bool = False
    use_rel_prompt: bool = False
    use_prefix_prompt_projection: bool = False
    use_rel_attn: bool = False
    use_length_embedding: bool = False
    use_rel_mention: bool = False
    use_filter_in_rel: Optional[str] = None
    use_rel_opt2: str = "mean"
    use_rel_opt3: Optional[str] = None
    use_ner_focal_loss: bool = False
    use_rel_focal_loss: bool = False
    use_ner_layer_loss: bool = True
    use_first_layer_loss: bool = True
    use_cross_ner: bool = False
    use_wider_ent_decode: bool = False
    use_normal_tag: bool = False
    use_spert: bool = False
    use_rel_loss_sum: Optional[int] = None
    use_dynamic_loss_sum: bool = False
    use_rel_na_warmup: Optional[str] = None
    use_warmup_rel: Optional[int] = None
    use_warmup_ner: Optional[int] = None
    use_warmup_filter: Optional[int] = None
    use_convert_warmup_to_steps: bool = False
    use_negative: Optional[str] = None
    use_filter_strategy: Optional[str] = None
    use_thres_threshold: float = 0.0
    use_train_threshold: bool = False
    use_thres_train: bool = False
    use_span_mention: bool = False
    use_two_plm: bool = False
    use_ner_hs: bool = False
    use_rel_ner: Optional[str] = None
    use_filter_loss_sum: bool = False
    use_semantic_loss: Optional[str] = None
    use_re_marker: bool = False
    norm_loss_bs: bool = False
    fix_pid: bool = False
    remove_ner_info: bool = False
    dry_test: bool = False

    # NER decoder
    ent_attn_layer_num: int = 3
    ent_attn_range: int = 0
    ent_mlp_layer_num: int = 2
    rel_mlp_layer_num: int = 2
    max_seq_len: int = 512
    context_window: int = 0
    max_span_length: int = 10
    max_tripes_count: int = 30
    data_piece: Optional[int] = None
    prompt_len: Optional[int] = None

    # Loss weights
    na_rel_weight: float = 1.0
    na_ner_weight: float = 1.0
    filter_dropout_rate: float = 0.1
    alb_test1: float = 0.5

    # Test from checkpoint
    test_from_ckpt: Optional[str] = None
    test_opt1: Optional[str] = None

    # Misc
    no_borther_confirm: bool = True
    fast_dev_run: bool = False

    model_config = {"extra": "allow"}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class FastEREConfig(BaseModel):
    """Full run configuration.  Load from YAML via ``FastEREConfig.from_yaml``."""

    env: EnvSettings = Field(default_factory=EnvSettings)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FastEREConfig":
        with open(path) as f:
            raw = f.read()

        env_data = {}
        env = EnvSettings(**env_data)

        # Substitute ${VAR} placeholders using env + EnvSettings values
        env_vars = {
            **os.environ,
            **{k.upper(): str(v) for k, v in env.model_dump().items()},
        }
        raw = re.sub(
            r"\$\{(\w+)\}", lambda m: env_vars.get(m.group(1), m.group(0)), raw
        )

        data = yaml.safe_load(raw)
        data.pop("env", None)

        model_data = data.pop("model", {})
        dataset_data = data.pop("dataset")
        training_data = data.pop("training", {})
        training_data.update(data)  # allow flat keys at top level for convenience

        return cls(
            env=env,
            model=ModelConfig(**model_data),
            dataset=DatasetConfig(**dataset_data),
            training=TrainingConfig(**training_data),
        )

    def to_runtime_config(self) -> "RuntimeConfig":
        """Convert to the dict-based RuntimeConfig used by model components."""
        return RuntimeConfig.from_fastere_config(self)


# ---------------------------------------------------------------------------
# RuntimeConfig — dict-based config compatible with existing model code
# ---------------------------------------------------------------------------


class _SimpleConfig(dict):
    """Case-insensitive dict with attribute access (legacy internal format)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def _key(self, key: str) -> str:
        return key.lower() if key else ""

    def __setattr__(self, key, value):
        self[self._key(key)] = value

    def __getattr__(self, key):
        return self.get(self._key(key))

    def __getitem__(self, key):
        return super().get(self._key(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._key(key), value)

    def __str__(self):
        return json.dumps(self)


class RuntimeConfig(_SimpleConfig):
    """Runtime configuration object used by Theta, DataModule, etc.

    Constructed from ``FastEREConfig`` via ``to_runtime_config()``.
    Exposes the same interface the existing model code expects.
    """

    @classmethod
    def from_fastere_config(cls, cfg: FastEREConfig) -> "RuntimeConfig":
        rc = cls()

        # Flatten training config into top-level keys
        for k, v in cfg.training.model_dump().items():
            rc[k] = v

        # Dataset / model as sub-dicts with attribute access
        rc["dataset"] = _SimpleConfig(cfg.dataset.model_dump())
        rc["model"] = _SimpleConfig(cfg.model.model_dump())

        # Paths from env
        rc["output"] = cfg.env.output_dir

        # Derived fields
        rc["model_config"] = cfg.model.model_name_or_path
        rc["dataset_config"] = ""  # resolved already
        rc["with_ext_config"] = False
        rc["args"] = Namespace()

        # Set up output directory
        tag = cfg.training.tag
        start = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        output_dir = os.path.join(
            cfg.env.output_dir, "training_output", f"{start}-{tag}"
        )
        if cfg.training.debug:
            output_dir = os.path.join(cfg.env.output_dir, "debug", f"{start}-{tag}")
        os.makedirs(output_dir, exist_ok=True)
        rc["output_dir"] = output_dir
        rc["test_result"] = os.path.join(output_dir, "test-result.json")
        rc["start"] = start

        # Load HuggingFace model config into rc.model
        try:
            from transformers import AutoConfig
            from transformers import logging as hf_logging

            hf_cfg = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
            rc["model"].update(hf_cfg.to_dict())
            if cfg.training.debug:
                hf_logging.set_verbosity_debug()
            else:
                hf_logging.set_verbosity_error()
        except Exception:
            pass

        # Precision
        try:
            import torch

            if rc["precision"] == 16:
                torch.set_float32_matmul_precision("medium")
            else:
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        return rc

    def save_config(self, filename: str = "config.yaml") -> None:
        config = {}
        for key, value in self.items():
            if isinstance(value, _SimpleConfig):
                config[key] = {
                    k: v
                    for k, v in value.items()
                    if not k.startswith("__") and not callable(v)
                }
            elif not key.startswith("__") and not callable(value) and key != "args":
                config[key] = value
        path = os.path.join(self["output_dir"], filename)
        with open(path, "w") as f:
            yaml.dump(config, f)
        self["final_config"] = path

    def save_best_model_path(self, path: str) -> None:
        if path:
            self["best_model_path"] = path
            self["last_model_path"] = path.replace(path.split(os.sep)[-1], "last.ckpt")

    def save_result(self, result: dict) -> None:
        with open(self["test_result"], "w") as f:
            json.dump(result, f, indent=4)
