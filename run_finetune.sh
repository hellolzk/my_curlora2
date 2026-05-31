#!/usr/bin/env bash

set -euo pipefail # 遇到错误立即退出，未定义变量时报错，管道中任一命令失败也算失败。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # 找到这个脚本所在目录，避免从别的目录启动时路径错乱。
cd "$SCRIPT_DIR" # 切换到脚本目录，这样后面的 finetune_unified.py 可以被稳定找到。

# source /path/to/miniconda3/bin/activate lzkenv # 如需指定 conda 环境，取消本行注释并改成你的环境路径。

export WANDB_DISABLED="true" # 关闭 Weights & Biases 日志上传，避免训练时弹登录或联网。
export SWANLAB_DISABLED="true" # 关闭 SwanLab 日志上传，保持训练只在本地输出日志。

MODEL_PATH="${MODEL_PATH:-/mnt/bn/chenhaobo-va-data/liuzekun2/models/Qwen3-4B}" # 基座模型目录；也可以运行前用环境变量 MODEL_PATH 覆盖。
DATASET_PATH="${DATASET_PATH:-/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2process_data/alpaca/train_alpaca.json}" # 训练数据 JSON 文件路径；默认使用 alpaca 训练集。
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2/curlora_adapter/qwen3_4b}" # 适配器和配置的输出根目录；脚本会在里面自动加数据集名和时间戳。

RANK_C="${RANK_C:-16}" # C 矩阵采样的列数；越大表达能力越强，但显存和计算更多。
RANK_R="${RANK_R:-16}" # R 矩阵采样的行数；通常和 RANK_C 保持一致，便于控制规模。
ALPHA="${ALPHA:-32}" # CUR-LoRA 分支的缩放系数；最终输出会加上 alpha 倍的 CUR 分支。
DROPOUT="${DROPOUT:-0.05}" # CUR 分支 dropout 概率；适当 dropout 可以减轻过拟合。
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-inverse}" # 采样策略：normal 按能量采样，inverse 偏向低能量行列，random 均匀随机。
U_INIT="${U_INIT:-zero}" # U 矩阵初始化方式：zero 初始不扰动原模型，kaiming 初始就有非零 CUR 分支。

TRAIN_C_FLAG="${TRAIN_C_FLAG:-}" # 是否训练 C；想训练 C 时设置为 --train_C，留空表示固定 C。
TRAIN_U_FLAG="${TRAIN_U_FLAG:---train_U}" # 是否训练 U；可设为 --no_train_U 来冻结 U 并用伪逆初始化。
TRAIN_R_FLAG="${TRAIN_R_FLAG:-}" # 是否训练 R；想训练 R 时设置为 --train_R，留空表示固定 R。
REPLACE_FLAG="${REPLACE_FLAG:---replace}" # 是否有放回采样；可设为 --no_replace 来禁止重复抽中同一行或同一列。
ADJUST_DUPS_FLAG="${ADJUST_DUPS_FLAG:---adjust_dups}" # 是否合并并缩放重复样本；可设为 --no_adjust_dups 来保留重复样本。

LR="${LR:-2e-4}" # 学习率；只训练少量 CUR 参数时通常可以比全量微调稍大。
EPOCHS="${EPOCHS:-3}" # 训练轮数；表示完整遍历训练集多少遍。
BATCH_SIZE="${BATCH_SIZE:-4}" # 单张设备每次前向/反向处理的样本数；显存不够时调小。
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}" # 梯度累积步数；有效 batch size = BATCH_SIZE * ACCUMULATION_STEPS * GPU 数量。
MAX_LEN="${MAX_LEN:-512}" # tokenizer 截断后的最大 token 长度；太长会更占显存。
WARMUP="${WARMUP:-0.1}" # warmup 比例；训练前 10% 步数逐渐升高学习率。
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}" # 权重衰减系数；用于正则化可训练参数。
LR_SCHEDULER="${LR_SCHEDULER:-cosine}" # 学习率调度器；必须是 finetune_unified.py 支持的取值。
SEED="${SEED:-42}" # 随机种子；会影响 CUR 采样、PyTorch 初始化和训练随机性。
OPTIM="${OPTIM:-adamw_torch}" # 优化器名称；传给 Transformers TrainingArguments。
SAVE_LIMIT="${SAVE_LIMIT:-2}" # 最多保留多少个 checkpoint；超过后自动删旧 checkpoint。
LOGGING_STEPS="${LOGGING_STEPS:-10}" # 每隔多少训练步打印一次日志。
SAVE_STEPS="${SAVE_STEPS:-100}" # 每隔多少训练步保存一次 checkpoint。
MAX_STEPS="${MAX_STEPS:--1}" # 最大训练步数；-1 表示按 EPOCHS 训练，不强行限制步数。
BF16_FLAG="${BF16_FLAG:---bf16}" # 是否使用 bfloat16；显卡支持时可省显存并加速。
GRADIENT_CHECKPOINTING_FLAG="${GRADIENT_CHECKPOINTING_FLAG:---gradient_checkpointing}" # 是否开启梯度检查点；省显存但训练会更慢。

CMD=( # 用数组保存命令，避免空字符串参数和空格路径导致解析错误。
  python # 使用当前环境里的 Python 解释器启动训练。
  finetune_unified.py # 训练入口脚本，负责加载模型、替换 CUR-LoRA 层并启动 Trainer。
  --model_path "$MODEL_PATH" # 告诉训练脚本基座模型在哪里。
  --dataset_path "$DATASET_PATH" # 告诉训练脚本训练数据在哪里。
  --output_base_dir "$OUTPUT_DIR" # 告诉训练脚本适配器保存到哪里。
  --rank_c "$RANK_C" # 传入 C 矩阵采样列数。
  --rank_r "$RANK_R" # 传入 R 矩阵采样行数。
  --alpha "$ALPHA" # 传入 CUR-LoRA 分支缩放系数。
  --dropout "$DROPOUT" # 传入 CUR 分支 dropout 概率。
  --sampling_strategy "$SAMPLING_STRATEGY" # 传入 CUR 行列采样策略。
  --u_init "$U_INIT" # 传入 U 矩阵初始化方式。
  --learning_rate "$LR" # 传入学习率。
  --num_train_epochs "$EPOCHS" # 传入训练轮数。
  --per_device_train_batch_size "$BATCH_SIZE" # 传入单卡 batch size。
  --gradient_accumulation_steps "$ACCUMULATION_STEPS" # 传入梯度累积步数。
  --max_length "$MAX_LEN" # 传入最大 token 长度。
  --warmup_ratio "$WARMUP" # 传入 warmup 比例。
  --weight_decay "$WEIGHT_DECAY" # 传入权重衰减系数。
  --lr_scheduler_type "$LR_SCHEDULER" # 传入学习率调度器类型。
  --seed "$SEED" # 传入随机种子，保证 CUR 采样尽量可复现。
  --optim "$OPTIM" # 传入优化器类型。
  --save_total_limit "$SAVE_LIMIT" # 传入 checkpoint 保留数量上限。
  --logging_steps "$LOGGING_STEPS" # 传入日志打印间隔。
  --save_steps "$SAVE_STEPS" # 传入 checkpoint 保存间隔。
  --max_steps "$MAX_STEPS" # 传入最大训练步数，-1 表示不覆盖 epoch 设置。
) # 命令数组定义结束。

if [[ -n "$TRAIN_C_FLAG" ]]; then # 如果用户设置了训练 C 的开关，就把它加入命令。
  CMD+=("$TRAIN_C_FLAG") # 添加 --train_C，让 C 矩阵也参与训练。
fi # 结束 TRAIN_C_FLAG 判断。

if [[ -n "$TRAIN_U_FLAG" ]]; then # 如果用户设置了训练 U 的开关，就把它加入命令。
  CMD+=("$TRAIN_U_FLAG") # 添加 --train_U，让 U 矩阵参与训练。
fi # 结束 TRAIN_U_FLAG 判断。

if [[ -n "$TRAIN_R_FLAG" ]]; then # 如果用户设置了训练 R 的开关，就把它加入命令。
  CMD+=("$TRAIN_R_FLAG") # 添加 --train_R，让 R 矩阵也参与训练。
fi # 结束 TRAIN_R_FLAG 判断。

if [[ -n "$REPLACE_FLAG" ]]; then # 如果用户设置了有放回采样开关，就把它加入命令。
  CMD+=("$REPLACE_FLAG") # 添加 --replace，允许重复抽中同一行或同一列。
fi # 结束 REPLACE_FLAG 判断。

if [[ -n "$ADJUST_DUPS_FLAG" ]]; then # 如果用户设置了重复样本调整开关，就把它加入命令。
  CMD+=("$ADJUST_DUPS_FLAG") # 添加 --adjust_dups，对重复采样的行列做合并和缩放。
fi # 结束 ADJUST_DUPS_FLAG 判断。

if [[ -n "$BF16_FLAG" ]]; then # 如果用户设置了 bf16 开关，就把它加入命令。
  CMD+=("$BF16_FLAG") # 添加 --bf16，使用 bfloat16 加载和训练模型。
fi # 结束 BF16_FLAG 判断。

if [[ -n "$GRADIENT_CHECKPOINTING_FLAG" ]]; then # 如果用户设置了梯度检查点开关，就把它加入命令。
  CMD+=("$GRADIENT_CHECKPOINTING_FLAG") # 添加 --gradient_checkpointing，用计算时间换更低显存占用。
fi # 结束 GRADIENT_CHECKPOINTING_FLAG 判断。

echo "开始 CUR-LoRA 微调，工作目录：$SCRIPT_DIR" # 打印当前工作目录，方便确认脚本运行位置。
echo "模型路径：$MODEL_PATH" # 打印模型路径，方便检查是否指向正确基座模型。
echo "数据路径：$DATASET_PATH" # 打印数据路径，方便检查训练集是否正确。
echo "输出目录：$OUTPUT_DIR" # 打印输出目录，方便训练后找到适配器。
printf '执行命令：%q ' "${CMD[@]}" # 以可复制的 shell 格式打印完整命令，方便排查参数。
printf '\n' # 打印换行，让后续训练日志从新行开始。

"${CMD[@]}" # 真正执行训练命令，数组展开可安全处理带空格的路径。

echo "CUR-LoRA 微调任务完成。" # 训练脚本正常结束后打印完成提示。
