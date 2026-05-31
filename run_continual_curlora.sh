#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export SWANLAB_DISABLED="${SWANLAB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/liuzekun2/models/Qwen3-4B}"
DATA_ROOT="${DATA_ROOT:-/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2process_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2/continual_runs/qwen3_4b_curlora}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

RANK_C="${RANK_C:-16}"
RANK_R="${RANK_R:-16}"
ALPHA="${ALPHA:-32}"
DROPOUT="${DROPOUT:-0.05}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-inverse}"
U_INIT="${U_INIT:-zero}"

LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
MAX_LEN="${MAX_LEN:-512}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
TRAIN_SAMPLE_RATIO="${TRAIN_SAMPLE_RATIO:-1.0}"
WARMUP="${WARMUP:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-200}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"
EVAL_SAMPLE_RATIO="${EVAL_SAMPLE_RATIO:-1.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

TASK_ORDER=(alpaca BoolQ MNLI MRPC QQP SIQA SST2)
HISTORY_TASKS=()
PREV_ADAPTER=""

mkdir -p "$OUTPUT_ROOT"
rm -f "$OUTPUT_ROOT/continual_metrics.jsonl"

echo "Model: $MODEL_PATH"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

for STAGE_INDEX in "${!TASK_ORDER[@]}"; do
  TASK_NAME="${TASK_ORDER[$STAGE_INDEX]}"
  STAGE_NAME="$(printf "%02d_%s" "$STAGE_INDEX" "$TASK_NAME")"
  STAGE_DIR="$OUTPUT_ROOT/$STAGE_NAME"

  echo "========== Train stage $STAGE_NAME =========="

  TRAIN_CMD=(
    torchrun
    --nproc_per_node "$NUM_GPUS"
    --master_port "$MASTER_PORT"
    continual_train.py
    --task_name "$TASK_NAME"
    --stage_index "$STAGE_INDEX"
    --model_path "$MODEL_PATH"
    --data_root "$DATA_ROOT"
    --output_root "$OUTPUT_ROOT"
    --rank_c "$RANK_C"
    --rank_r "$RANK_R"
    --alpha "$ALPHA"
    --dropout "$DROPOUT"
    --sampling_strategy "$SAMPLING_STRATEGY"
    --u_init "$U_INIT"
    --learning_rate "$LR"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$ACCUMULATION_STEPS"
    --max_length "$MAX_LEN"
    --max_train_samples "$MAX_TRAIN_SAMPLES"
    --train_sample_ratio "$TRAIN_SAMPLE_RATIO"
    --warmup_ratio "$WARMUP"
    --weight_decay "$WEIGHT_DECAY"
    --seed "$SEED"
    --save_steps "$SAVE_STEPS"
    --bf16
    --gradient_checkpointing
    --overwrite_output_dir
  )

  if [[ -n "$PREV_ADAPTER" ]]; then
    TRAIN_CMD+=(--init_adapter_path "$PREV_ADAPTER")
  fi

  "${TRAIN_CMD[@]}"

  PREV_ADAPTER="$STAGE_DIR/curlora_adapter.bin"
  HISTORY_TASKS+=("$TASK_NAME")

  echo "========== Evaluate stage $STAGE_NAME on history tasks: ${HISTORY_TASKS[*]} =========="

  python continual_eval.py \
    --adapter_path "$PREV_ADAPTER" \
    --stage_name "$STAGE_NAME" \
    --tasks "${HISTORY_TASKS[@]}" \
    --model_path "$MODEL_PATH" \
    --data_root "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --rank_c "$RANK_C" \
    --rank_r "$RANK_R" \
    --alpha "$ALPHA" \
    --dropout 0.0 \
    --sampling_strategy "$SAMPLING_STRATEGY" \
    --u_init "$U_INIT" \
    --seed "$SEED" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --eval_sample_ratio "$EVAL_SAMPLE_RATIO" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --bf16
done

python summarize_forgetting.py --metrics_jsonl "$OUTPUT_ROOT/continual_metrics.jsonl" --output_dir "$OUTPUT_ROOT"

echo "Continual CUR-LoRA experiment finished: $OUTPUT_ROOT"
