#!/bin/bash

# --- 1. 环境配置 ---
# 激活 conda 环境（根据你的实际环境路径）
# source /root/shared-nvme/miniconda3/bin/activate lzkenv

# 设置工作目录
cd /root/shared-nvme/essence_of_lora/my_curlora

# 禁用不需要的日志库
export WANDB_DISABLED="true"
export SWANLAB_DISABLED="true"

# --- 2. 路径配置 ---
MODEL_PATH="/root/shared-nvme/essence_of_lora/my_curlora/model/Qwen3-4B/Qwen/Qwen3-4B"
DATASET_PATH="/root/shared-nvme/essence_of_lora/my_curlora/process_data/alpaca/train_alpaca.json"
OUTPUT_DIR="/root/shared-nvme/essence_of_lora/my_curlora/curlora_adapter/qwen3_4b"

# --- 3. CUR 超参数 ---
RANK_C=16
RANK_R=16
ALPHA=32
DROPOUT=0.05
SAMPLING_STRATEGY="normal" # normal, inverse, random
U_INIT="zero"              # zero, kaiming

# 矩阵训练控制 (若要训练请设为 "--train_X"，不训练留空)
TRAIN_C=""                 # 例如 "--train_C"
TRAIN_U="--train_U"        # 默认训练 U
TRAIN_R=""                 # 例如 "--train_R"

# 采样设置
REPLACE="--replace"        # 是否有放回抽样
ADJUST_DUPS="--adjust_dups" # 是否开启重复项调整

# --- 4. 训练超参数 ---
LR=2e-4
EPOCHS=3
BATCH_SIZE=4
ACCUMULATION_STEPS=4
MAX_LEN=512
WARMUP=0.1
WEIGHT_DECAY=0.01
LR_SCHEDULER="cosine"      # linear, cosine, etc.
SEED=42
OPTIM="adamw_torch"
SAVE_LIMIT=2
BF16="--bf16"              # 硬件支持建议开启
GRADIENT_CHECKPOINTING="--gradient_checkpointing" # 显存不足建议开启

# --- 5. 执行训练 ---
echo "🚀 开始 CUR-LoRA 微调..."
python finetune_unified.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_base_dir "$OUTPUT_DIR" \
    --rank_c $RANK_C \
    --rank_r $RANK_R \
    --alpha $ALPHA \
    --dropout $DROPOUT \
    $TRAIN_C \
    $TRAIN_U \
    $TRAIN_R \
    --sampling_strategy "$SAMPLING_STRATEGY" \
    $REPLACE \
    $ADJUST_DUPS \
    --u_init "$U_INIT" \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION_STEPS \
    --max_length $MAX_LEN \
    --warmup_ratio $WARMUP \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --seed $SEED \
    --optim "$OPTIM" \
    --save_total_limit $SAVE_LIMIT \
    $BF16 \
    $GRADIENT_CHECKPOINTING

echo "✅ 微调任务完成！"
