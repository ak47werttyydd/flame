export HF_HOME=/home/a84400789/.cache/data/huggingface
export PYTHONHTTPSVERIFY=0
export WANDB_MODE=offline
export SENTRY_DSN=""

BASE_DUMP_PREFIX="exp/gdn-340M-4K-20B/lr_search"
MODEL_REPO="m-a-p/340M-20B-GatedDeltaNet-hybrid-3-1"
DATASET_DIR="/home/a84400789/.cache/huggingface/datasets/fineweb-edu-100_bt/default/0.0.0/26426dc754e0ed07"

NGPU=4
SEQ_LEN=4096
BATCH_SIZE=13
GRAD_ACCUM=2
TOTAL_TOKENS=20000000000

# global_batch_tokens = batch_size * seq_len * ngpu * grad_accum = 13 * 4096 * 4 * 2 = 426,496 tokens
REAL_GLOBAL_BATCH_TOKENS=$((BATCH_SIZE * SEQ_LEN * NGPU * GRAD_ACCUM))
STEPS=$((TOTAL_TOKENS / REAL_GLOBAL_BATCH_TOKENS))

LR_LIST=(3e-4 4e-4 7e-4 1e-3 2.15e-3 4.3e-3)

for LR in "${LR_LIST[@]}"; do
  DUMP_FOLDER="${BASE_DUMP_PREFIX}.ngpu${NGPU}.bs${BATCH_SIZE}.ga${GRAD_ACCUM}.steps${STEPS}.lr${LR}"
  LOG_FILE="${DUMP_FOLDER}.log"

  echo "===== Running LR=${LR} (BS=${BATCH_SIZE}, GA=${GRAD_ACCUM}, Steps=${STEPS}) -> ${DUMP_FOLDER} ====="

  CUDA_VISIBLE_DEVICES=0,1,2,3 NNODE=1 NGPU=${NGPU} LOG_RANK=0 bash /home/a84400789/flame/train.sh \
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
    --training.batch_size ${BATCH_SIZE} \
    --training.seq_len ${SEQ_LEN} \
    --training.context_len ${SEQ_LEN} \
    --training.gradient_accumulation_steps ${GRAD_ACCUM} \
    --training.steps ${STEPS} \
    --training.max_norm 1.0 \
    --training.skip_nan_inf \
    --training.dataset "${DATASET_DIR}" \
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
