# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Flame** is a minimal, efficient distributed LLM training framework built on PyTorch and [TorchTitan](https://github.com/pytorch/torchtitan), with deep integration with [Flash-Linear-Attention (FLA)](https://github.com/fla-org/flash-linear-attention) architectures. It supports Transformer, GLA, Mamba2, HGRN2, GSA, DeltaNet, GatedDeltaNet, and hybrid model variants.

## Key Commands

**Install:**
```bash
pip install .
pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention
pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c
```

**Train (single node):**
```bash
NGPU=8 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --model.config configs/transformer_340M.json \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.steps 20480
```

**Train (single GPU for debugging):**
```bash
NGPU=1 bash train.sh --job.config_file flame/models/fla.toml ...
```

**Multi-node training:**
```bash
NNODE=<num_nodes> MASTER_ADDR=<addr> MASTER_PORT=<port> bash ParallelTrain4RankPerNode.sh ...
```

**Convert checkpoint (DCP → HuggingFace):**
```bash
python -m flame.utils.convert_dcp_to_hf --path <checkpoint_dir> --step <step> --config <config.json> --tokenizer <tokenizer_name>
```

**Convert checkpoint (HuggingFace → DCP, for continual training):**
```bash
python -m flame.utils.convert_hf_to_dcp --path <hf_model_dir> --output <output_dir>
```

**Lint (pre-commit hooks: isort + flake8):**
```bash
pre-commit run --all-files
```
Max line length is **127 characters** (configured in `.flake8`).

## Architecture

### Core Training Pipeline

- **`flame/train.py`** — Entry point. Registers the FLA train spec with TorchTitan, initializes distributed training, manages the main training loop with checkpoint save/load, metrics (TensorBoard, W&B), and gradient accumulation.
- **`flame/config_manager.py`** — `JobConfig` class handles hierarchical configuration: TOML files + CLI args with precedence (CLI > TOML > defaults). Covers 40+ config sections (job, model, optimizer, training, checkpoint, profiling, etc.).
- **`flame/data.py`** — `BufferShuffledIterableDataset` for streaming data with online tokenization, weighted multi-dataset sampling, variable-length sequence packing (no padding), and stateful loading for resuming.

### Model Parallelism

- **`flame/models/parallelize_fla.py`** — Applies tensor parallelism (colwise/rowwise), activation checkpointing, and `torch.compile` to FLA models.
- **`flame/models/pipeline_fla.py`** — Pipeline parallelism with configurable split points and schedule types.
- **`flame/models/activation_offloading.py`** — CPU offloading for activation memory reduction.

Supported parallelism: **FSDP** (data), **Tensor**, **Pipeline**, **Context** (sequence), and **HSDP** (hybrid sharded).

### Configuration Files

JSON model configs in `configs/` define architecture hyperparameters (hidden size, num layers, heads, etc.) for each model family. The TOML file (`flame/models/fla.toml`) sets training defaults.

### Custom Models

Add new model architectures under `custom_models/`. Each must:
1. Implement a config class inheriting from `PretrainedConfig` and a model class inheriting from `PreTrainedModel`.
2. Register with `AutoModelForCausalLM`.
3. Follow the pattern in `custom_models/sba/` as a reference.

### Utilities

- `flame/utils/check_dcp_nan.py`, `check_conv_nan.py` — Debug NaN/Inf in checkpoints and conversions.
- `flame/utils/preprocess.py` — Offline data preprocessing.
- `critical_batch_size.py` — Batch size analysis tool.

## Key Configuration Options

| Parameter | Description |
|---|---|
| `training.varlen` | Enable variable-length sequence packing (no padding) |
| `training.context_len` | Max context length (can differ from `seq_len`) |
| `training.gradient_accumulation_steps` | Accumulate gradients before optimizer step |
| `training.compile` | Enable `torch.compile` (not compatible with all FLA kernels) |
| `checkpoint.async_mode` | Async checkpoint saving (`"async"` or `"disabled"`) |
| `training.skip_nan_inf` | Skip optimizer step if grad norm is NaN/Inf |

## Dependencies & Submodules

Three git submodules in `3rdparty/`:
- `flash-linear-attention` — FLA kernels (Triton-based)
- `torchtitan` — PyTorch distributed training base framework
- `lm-evaluation-harness` — Model evaluation

Requires: Python ≥ 3.10, PyTorch ≥ 2.5, `transformers < 5.0.0` (required for weight conversion compatibility), Triton ≥ 3.0.

## Docker

Docker setup in `docker/`. Base image: `nvcr.io/nvidia/pytorch:25.06-py3` (CUDA 12.9.1, PyTorch 2.8.x, Python 3.12). Requires driver ≥ 570 for consumer GPUs or ≥ 535.86 for datacenter GPUs.
