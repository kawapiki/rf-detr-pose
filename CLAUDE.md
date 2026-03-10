# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Setup (always use --all-groups for full dev environment)
uv sync --all-groups

# Run CPU tests (matches CI)
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml

# Run a single test file or test
uv run --no-sync pytest tests/models/test_config.py -v
uv run --no-sync pytest tests/models/test_config.py::TestClassName::test_method -v

# Run GPU tests
uv run --no-sync pytest src/ tests/ -n 2 -m gpu

# Lint/format (ALWAYS run before committing)
pre-commit run --all-files

# Build package
uv build
```

## Architecture Overview

RF-DETR is a real-time object detection and instance segmentation library built on DINOv2 + deformable DETR.

### Three-Layer Model Architecture

**Layer 1 — User-facing wrappers** (`src/rfdetr/detr.py`): `RFDETR` base class and size variants (`RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`, `RFDETRSeg*`). These are plain Python objects (not `nn.Module`). Each variant overrides `get_model_config()` and `get_train_config()`. Public API: `.predict()`, `.train()`, `.export()`.

**Layer 2 — Orchestration** (`src/rfdetr/main.py`): `Model` class (also not `nn.Module`). Owns `self.model` (the PyTorch module) and `self.postprocess`. Handles weight loading, LoRA application, training loop orchestration, and export. The `populate_args()` function bridges Pydantic configs → `argparse.Namespace` consumed by internal functions.

**Layer 3 — PyTorch module** (`src/rfdetr/models/lwdetr.py`): `LWDETR(nn.Module)` — the actual neural network. Contains `backbone` (DinoV2 + MultiScaleProjector), `transformer` (MSDeformAttn decoder), classification/regression heads, and optional `segmentation_head`. Forward returns dict with `pred_logits`, `pred_boxes`, optionally `pred_masks`, `aux_outputs`.

### Key relationships

- `RFDETR.model` → `main.Model` instance
- `main.Model.model` → `LWDETR` nn.Module (the actual PyTorch model)
- Public classes are exported from `src/rfdetr/__init__.py` (imports from `detr.py`)
- Plus models (`RFDETRXLarge`, `RFDETR2XLarge`) are lazily imported from `rfdetr_plus` package via `__getattr__` in `__init__.py` and `platform/models.py`

### Configuration System (`src/rfdetr/config.py`)

All configs are **Pydantic v2** `BaseModel` with `extra="forbid"` (typos in kwargs raise `ValidationError`).

- `ModelConfig` → variant-specific subclasses (`RFDETRBaseConfig`, `RFDETRNanoConfig`, etc.) define architecture params: encoder, hidden_dim, dec_layers, resolution, num_queries, etc.
- `TrainConfig` / `SegmentationTrainConfig` → training hyperparameters: lr, batch_size, epochs, dataset paths, augmentation, metric sinks.
- Config merging in `train_from_config()`: model_config values fill in where train_config has `None`.

### Training Flow

`RFDETR.train()` → `train_from_config()` → `Model.train()` → loops over epochs calling `train_one_epoch()` / `evaluate()` from `engine.py`. Datasets are built by `datasets/__init__.py:build_dataset()` (supports COCO, YOLO, Objects365 formats). Callbacks dict (`on_fit_epoch_end`, `on_train_end`) drives metric sinks.

### Inference Flow

`RFDETR.predict()` normalizes images → resizes to model resolution → runs `LWDETR.forward()` → `PostProcess` applies sigmoid + top-K + box conversion → filters by threshold → returns `supervision.Detections`.

### Other Key Modules

- `src/rfdetr/datasets/` — Dataset loaders, transforms, augmentation config, COCO evaluator, synthetic dataset generator for tests
- `src/rfdetr/assets/model_weights.py` — `ModelWeights` enum, weight download/validation
- `src/rfdetr/util/misc.py` — `NestedTensor` (tensors + padding mask), `collate_fn`, distributed utilities
- `src/rfdetr/deploy/` — ONNX export and benchmarking
- `src/rfdetr/cli/main.py` — CLI entry point (`rfdetr` command), thin argparse wrapper around `RFDETRBase.train()`

## Code Conventions

- **Imports**: Always direct — `from rfdetr.util.misc import get_rank` (never `import ... as`)
- **Logger**: `from rfdetr.util.logger import get_logger; logger = get_logger()` — reads `LOG_LEVEL` env var
- **TQDM**: `from tqdm.auto import tqdm` (not `from tqdm import tqdm`)
- **Type hints and Google-style docstrings** are mandatory for all functions and classes. Do not duplicate types in docstrings.
- **License header** required on all Python files (enforced by pre-commit):
  ```python
  # ------------------------------------------------------------------------
  # RF-DETR
  # Copyright (c) 2025 Roboflow. All Rights Reserved.
  # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
  # ------------------------------------------------------------------------
  ```
- **Line length**: 120 characters (ruff)
- **Target Python**: 3.10+

## Testing Conventions

- Use TDD: bug fixes get a failing test first, features get comprehensive tests
- Group tests in classes, use `@pytest.mark.parametrize` with `pytest.param(..., id="name")`
- Mark GPU tests with `@pytest.mark.gpu`
- `tests/conftest.py` provides `synthetic_shape_dataset_dir` fixture (generates COCO-format dataset for integration tests)
- Avoid multiple validation cases in a single test — split into parametrized cases
- `--doctest-modules` is enabled in pytest config — doctests in source files run automatically

## Branch Conventions

Branch naming: `{type}/{issue_number}-description` (e.g., `fix/123-authentication_bug`, `feat/678-add_export_support`)

Prefixes: `fix/`, `feat/`, `docs/`, `refactor/`, `test/`, `chore/`
