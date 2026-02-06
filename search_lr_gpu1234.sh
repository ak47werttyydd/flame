export HF_HOME=/home/a84400789/.cache/data/huggingface
export PYTHONHTTPSVERIFY=0
export WANDB_MODE=offline
export SENTRY_DSN=""

BASE_DUMP_PREFIX="exp/gdn-340M-4K-20B/batch1.seqlen4096.context4096.warmup1024.update1.steps135633"
MODEL_REPO="m-a-p/340M-20B-GatedDeltaNet-pure-baseline"
DATASET_DIR="/home/a84400789/.cache/data/fineweb-edu-100BT"

LRS=(1e-4 6e-4)

for LR in "${LRS[@]}"; do
  DUMP_FOLDER="${BASE_DUMP_PREFIX}.lr${LR}.cosine"
  LOG_FILE="${DUMP_FOLDER}.log"

  echo "===== Running LR=${LR} -> ${DUMP_FOLDER} ====="

  CUDA_VISIBLE_DEVICES=0,1,2,3 NNODE=1 NGPU=4 LOG_RANK=0 bash /home/a84400789/flame/train.sh \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder "${DUMP_FOLDER}" \
    --model.config "${MODEL_REPO}" \
    --model.tokenizer_path "${MODEL_REPO}" \
    --optimizer.name AdamW \
    --optimizer.eps 1e-15 \
    --optimizer.lr "${LR}" \
    --lr_scheduler.warmup_steps 1024 \
    --lr_scheduler.lr_min 0.1 \
    --lr_scheduler.decay_type cosine \
    --training.batch_size 9 \
    --training.seq_len 4096 \
    --training.context_len 4096 \
    --training.gradient_accumulation_steps 1 \
    --training.steps 135633 \
    --training.max_norm 1.0 \
    --training.skip_nan_inf \
    --training.dataset "${DATASET_DIR}" \
    --training.dataset_name default \
    --training.dataset_split train \
    --training.num_workers 32 \
    --training.prefetch_factor 2 \
    --training.seed 42 \
    --checkpoint.interval 20480 \
    --checkpoint.load_step 0 \
    --checkpoint.keep_latest_k 2 \
    --metrics.log_freq 1 \
    2>&1 | tee "${LOG_FILE}"

  RC=${PIPESTATUS[0]}
  if [ "$RC" -ne 0 ]; then
    echo "!!!!! LR=${LR} failed with exit code ${RC}. Continuing to next LR..."
  else
    echo "===== LR=${LR} finished OK ====="
  fi
done