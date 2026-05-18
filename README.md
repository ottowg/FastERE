# FastERE

An updated fork of [FastERE](https://github.com/wjw136/FastERE), a fast framework for joint **Entity and Relation Extraction** (ERE).

## What it does

FastERE jointly extracts named entities and relations between them from text in a single model pass. It uses pre-trained language models (BERT, RoBERTa, ALBERT, SciBERT) as encoders and adds three decoder heads:

1. **NER decoder** — identifies entity spans and their types (e.g. Person, Organization) using a span-based attention model with BIO tagging.
2. **Entity-pair filter** — prunes candidate entity pairs before relation classification, improving efficiency and precision.
3. **Relation decoder** — classifies the relation type between each surviving entity pair.

The model is trained end-to-end with a combined NER + filter + relation loss.

## Paper

If you use this code, please cite the original paper:

> Wang, J., Jiang, J., & Zhang, M. (2025). **FastERE: A Fast Framework for Entity Relation Extraction**.
> *Data Mining and Knowledge Discovery*.
> https://dl.acm.org/doi/abs/10.1007/s10618-025-01146-y

## Changes from the original

This fork modernises the codebase:

- Restructured as an installable `src/fastere/` package
- Package management via **uv**
- Pydantic v2 config loaded from a YAML file; project-wide paths via `.env`
- CLI entry points (`fastere-train`, `fastere-evaluate`) replacing the old `main.py`
- Updated dependencies (PyTorch 2.x, PyTorch Lightning 2.x, Transformers 4.48+)
- Test suite with pytest

## Installation

```bash
git clone <this-repo>
cd FastERE

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set your dataset path:

```bash
cp .env.example .env
# edit .env and set DATASET_DIR=/path/to/your/datasets
```

## Usage

```bash
# Train on SciERC
fastere-train --config configs/example_scierc.yaml

# Override individual settings
fastere-train --config configs/example_scierc.yaml --tag my-run --lr 2e-5 --max-epochs 50

# Evaluate a checkpoint
fastere-evaluate \
  --config configs/example_scierc.yaml \
  --checkpoint output/training_output/<run>/checkpoints/<ckpt>.ckpt \
  --output results.json
```

Set `CUDA_VISIBLE_DEVICES` to choose the GPU:

```bash
CUDA_VISIBLE_DEVICES=0 fastere-train --config configs/example_scierc.yaml
```

## Data format

Data must be in [DYGIE++](https://github.com/dwadden/dygiepp) JSON-lines format, one document per line:

```json
{"doc_key": "doc1", "sentences": [["The", "CEO", "joined"]], "ner": [[[0, 1, "PER"]]], "relations": [[[0, 1, 2, 3, "WORKS-FOR"]]]}
```

See `configs/example_scierc.yaml` and `configs/example_ace2005.yaml` for full configuration examples.

## Development

```bash
# Run tests
uv run pytest

# Lint and format (run after every change)
uv run ruff check src/ tests/ --select I --fix && uv run ruff format src/ tests/
```
