export HF_HOME=/home/a84400789/.cache/data/huggingface
export PYTHONHTTPSVERIFY=0
export WANDB_MODE=offline
export SENTRY_DSN=""

BASE_DUMP_PREFIX="exp/gdn-340M-4K-20B/batch_search.lr3e-4"
MODEL_REPO="m-a-p/340M-20B-GatedDeltaNet-pure-baseline"
DATASET_DIR="/home/a84400789/.cache/data/fineweb-edu-100BT"

LR=3e-4
NGPU=8
SEQ_LEN=4096
TOTAL_TOKENS=20000000000
MAX_BATCH_SIZE=17

# Global batch sizes in tokens
GLOBAL_BATCH_SIZES=(50000 196000 393000 442000 753000 3000000)

for GLOBAL_BATCH_TOKENS in "${GLOBAL_BATCH_SIZES[@]}"; do
  # Calculate per-GPU batch size and gradient accumulation steps
  # global_batch_tokens = batch_size * seq_len * ngpu * grad_accum_steps
  # We need: batch_size * grad_accum_steps = global_batch_tokens / (seq_len * ngpu)
  
  TOTAL_SAMPLES_PER_STEP=$((GLOBAL_BATCH_TOKENS / SEQ_LEN / NGPU))
  
  # Try to use largest possible batch_size that fits in memory
  # For H100, you might be able to use batch_size up to 16-32 depending on model size
  # Adjust MAX_BATCH_SIZE based on your memory constraints
  
  if [ $TOTAL_SAMPLES_PER_STEP -le $MAX_BATCH_SIZE ]; then
    BATCH_SIZE=$TOTAL_SAMPLES_PER_STEP
    GRAD_ACCUM=1
  else
    BATCH_SIZE=$MAX_BATCH_SIZE
    GRAD_ACCUM=$((TOTAL_SAMPLES_PER_STEP / BATCH_SIZE))
  fi
  
  # Calculate training steps for 20B tokens
  STEPS=$((TOTAL_TOKENS / GLOBAL_BATCH_TOKENS))
  
  DUMP_FOLDER="${BASE_DUMP_PREFIX}.gbs${GLOBAL_BATCH_TOKENS}.bs${BATCH_SIZE}.ga${GRAD_ACCUM}.steps${STEPS}"
  LOG_FILE="${DUMP_FOLDER}.log"

  echo "===== Running Global Batch=${GLOBAL_BATCH_TOKENS} tokens (BS=${BATCH_SIZE}, GA=${GRAD_ACCUM}, Steps=${STEPS}) -> ${DUMP_FOLDER} ====="

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NNODE=1 NGPU=${NGPU} LOG_RANK=0 bash /home/a84400789/flame/train.sh \
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
    echo "!!!!! Global Batch=${GLOBAL_BATCH_TOKENS} failed with exit code ${RC}. Continuing to next batch size..."
  else
    echo "===== Global Batch=${GLOBAL_BATCH_TOKENS} finished OK ====="
  fi
done