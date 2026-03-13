#!/usr/bin/bash
# ============================================================================
# Train Qwen3.5-35B-A3B (MoE) on 4 GPUs with FSDP
#
# Model: 35B total params, ~3B activated per token
#   - 40 layers, hybrid GatedDeltaNet + Full Attention (3:1)
#   - 256 routed experts (top-8) + 1 shared expert per layer
#
# Usage:
#   # Inside the Qwen3.5 docker container:
#   cd /home/a84400789/flame
#   bash train_qwen35_4gpu.sh
# ============================================================================

export HF_HOME=/home/.cache/huggingface
export WANDB_MODE=offline
export SENTRY_DSN=""

MODEL_REPO="Qwen/Qwen3.5-35B-A3B"
DATASET="HuggingFaceFW/fineweb-edu"

# Training hyperparameters
LR="1e-5"
WARMUP=100
STEPS=1000
BATCH_SIZE=1
SEQ_LEN=2048
CONTEXT_LEN=2048
GA_STEPS=4

DUMP_FOLDER="exp/qwen35-moe-a3b/bs${BATCH_SIZE}.ga${GA_STEPS}.seq${SEQ_LEN}.lr${LR}.steps${STEPS}"

echo "===== Training Qwen3.5-35B-A3B on 4 GPUs ====="
echo "  Effective batch: ${BATCH_SIZE} * 4 GPUs * ${GA_STEPS} GA = $((BATCH_SIZE * 4 * GA_STEPS)) sequences/step"
echo "  Tokens per step: $((BATCH_SIZE * 4 * GA_STEPS * SEQ_LEN))"
echo "  Dump folder: ${DUMP_FOLDER}"

NGPU=4 NNODE=1 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/qwen3_5_moe.toml \
  --job.dump_folder "${DUMP_FOLDER}" \
  --model.config "${MODEL_REPO}" \
  --model.tokenizer_path "${MODEL_REPO}" \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr "${LR}" \
  --lr_scheduler.warmup_steps ${WARMUP} \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size ${BATCH_SIZE} \
  --training.seq_len ${SEQ_LEN} \
  --training.context_len ${CONTEXT_LEN} \
  --training.gradient_accumulation_steps ${GA_STEPS} \
  --training.steps ${STEPS} \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset "${DATASET}" \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 4 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.data_parallel_shard_degree -1 \
  --training.tensor_parallel_degree 1 \
  --activation_checkpoint.mode full \
  --checkpoint.interval 500 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10
