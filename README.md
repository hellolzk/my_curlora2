# CUR-LoRA Continual Learning Forgetting Benchmark

本工程用于研究 CUR-LoRA 在连续任务微调中的遗忘问题。实验流程是：

1. 按固定顺序依次训练 7 个任务。
2. 每训练完一个任务，立即评测当前任务和所有历史任务。
3. 记录每个阶段的 adapter、预测结果、任务指标和遗忘汇总。

默认任务顺序：

```text
alpaca -> BoolQ -> MNLI -> MRPC -> QQP -> SIQA -> SST2
```

## 目录结构

核心文件：

```text
my_curlora/
  unified_curlora.py             # CUR-LoRA 模块实现
  curlora_utils.py               # CUR-LoRA 注入、adapter 读写、SFT 数据处理工具
  continual_config.py            # 7 个任务的路径、顺序和评测配置
  continual_train.py             # 单个 stage 的持续微调脚本
  continual_eval.py              # 单个 stage 后的历史任务评测脚本
  summarize_forgetting.py        # 根据 metrics JSONL 汇总遗忘指标
  run_continual_curlora.sh       # 8 卡 H100 一键连续训练和评测脚本
  requirements.txt               # Python 依赖
```

默认输入路径：

```text
模型路径：
/Users/bytedance/essential_of_lora/model/models/Qwen3-4B

数据路径：
/Users/bytedance/essential_of_lora/process_data/process_data
```

默认输出路径：

```text
/Users/bytedance/essential_of_lora/my_curlora/continual_runs/qwen3_4b_curlora
```

## 环境要求

硬件：

```text
8 x H100
CUDA 12.4
```

建议 Python：

```text
Python 3.10 或 Python 3.11
```

安装依赖：

```bash
cd /Users/bytedance/essential_of_lora/my_curlora
pip install -r requirements.txt
```

如果你的环境需要手动指定 CUDA 12.4 的 PyTorch wheel，可以先安装 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

如果 `vllm` 安装和本机 CUDA/PyTorch 版本冲突，可以先跳过：

```bash
grep -v '^vllm' requirements.txt > requirements_no_vllm.txt
pip install -r requirements_no_vllm.txt
```

当前持续评测脚本使用 Hugging Face `transformers.generate()`，不强依赖 `vllm`。

## 数据格式

7 个任务都应是 JSON list，且每条样本包含：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

默认读取路径在 `continual_config.py` 中配置：

```text
alpaca: alpaca/train_alpaca.json, alpaca/test_alpaca.json
BoolQ: BoolQ/boolq_processed/train.json, BoolQ/boolq_processed/validation.json
MNLI: MNLI/mnli_processed/train_new_mnli.json, MNLI/mnli_processed/test_new_mnil.json
MRPC: MRPC/mrpc_processed/train.json, MRPC/mrpc_processed/validation.json
QQP: QQP/qqp_processed/train.json, QQP/qqp_processed/validation.json
SIQA: SIQA/siqa_processed/train.json, SIQA/siqa_processed/validation.json
SST2: SST2/sst2_processed/train.json, SST2/sst2_processed/validation.json
```

## 抽样策略

为了避免 QQP、SST2、SIQA 这类大数据集在持续微调中主导参数更新，工程默认不对所有任务使用全量训练集，而是按任务规模做上限控制。

当前数据规模：

```text
任务      训练集条数    测试/验证条数
alpaca   5000        5000
BoolQ    9427        3270
MNLI     5000        5000
MRPC     3668        408
QQP      363846      40430
SIQA     33410       1954
SST2     67349       872
```

默认训练抽样：

```text
每个任务最多 5000 条训练样本
MRPC 只有 3668 条，因此使用全量
alpaca 和 MNLI 本身是 5000 条，因此使用全量
BoolQ、QQP、SIQA、SST2 从全量训练集中抽样到 5000 条
```

默认评测抽样：

```text
每个任务最多 1000 条评测样本
MRPC 验证集只有 408 条，因此使用全量
SST2 验证集只有 872 条，因此使用全量
其他任务默认抽样 1000 条
```

分类任务默认使用分层抽样：

```text
BoolQ: 按 yes / no 比例抽样
MNLI: 按 entailment / neutral / contradiction 比例抽样
MRPC: 按 equivalent / not equivalent 比例抽样
QQP: 按 duplicate / not_duplicate 比例抽样
SIQA: 按 A / B / C 答案前缀比例抽样
SST2: 按 positive / negative 比例抽样
```

Alpaca 是生成任务，没有固定类别，默认使用随机抽样。

每个训练 stage 的采样结果会写入：

```text
{OUTPUT_ROOT}/{stage_name}/stage_config.json
```

每个评测任务的采样结果会写入：

```text
{OUTPUT_ROOT}/{stage_name}/eval/{task_name}_metrics.json
```

## 一键运行

```bash
cd /Users/bytedance/essential_of_lora/my_curlora
bash run_continual_curlora.sh
```

脚本默认使用 8 张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
```

如果开发机路径不同，可以运行时覆盖：

```bash
MODEL_PATH=/mnt/bn/chenhaobo-va-data/liuzekun2/model/models/Qwen3-4B \
DATA_ROOT=/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2process_data \
OUTPUT_ROOT=/mnt/bn/chenhaobo-va-data/liuzekun2/curlora_forgetting_runs/qwen3_4b \
bash run_continual_curlora.sh
```

## 实验逻辑

第 0 阶段：

```text
训练 alpaca
评测 alpaca
保存 00_alpaca/curlora_adapter.bin
```

第 1 阶段：

```text
加载 00_alpaca/curlora_adapter.bin
继续训练 BoolQ
评测 alpaca 和 BoolQ
保存 01_BoolQ/curlora_adapter.bin
```

第 2 阶段：

```text
加载 01_BoolQ/curlora_adapter.bin
继续训练 MNLI
评测 alpaca、BoolQ、MNLI
保存 02_MNLI/curlora_adapter.bin
```

后续阶段以此类推。

注意：这里的持续学习是“同一个 CUR-LoRA adapter 持续更新”，不是每个任务单独保存一个互不相关的新 adapter。

## 输出说明

输出根目录示例：

```text
continual_runs/qwen3_4b_curlora/
  00_alpaca/
    curlora_adapter.bin
    stage_config.json
    adapter_path.txt
    checkpoints/
    eval/
      alpaca_predictions.json
      alpaca_metrics.json
      metrics.json
  01_BoolQ/
    curlora_adapter.bin
    stage_config.json
    eval/
      alpaca_predictions.json
      alpaca_metrics.json
      BoolQ_predictions.json
      BoolQ_metrics.json
      metrics.json
  continual_metrics.jsonl
  forgetting_summary.json
  forgetting_summary.csv
```

`continual_metrics.jsonl` 每行是一条阶段-任务指标：

```json
{"stage": "02_MNLI", "task": "BoolQ", "num_samples": 1000, "exact_match": 0.71, "rougeL": 0.72, "label_accuracy": 0.71}
```

`forgetting_summary.csv` 中的关键字段：

```text
first_score: 任务首次被训练后立刻评测的分数
best_score: 该任务在整个持续学习过程中的最好分数
final_score: 最后一个阶段后该任务的分数
forgetting_best_minus_final: best_score - final_score
drop_first_minus_final: first_score - final_score
```

## 评测指标

分类任务：

```text
BoolQ: label_accuracy, exact_match, rougeL
MNLI: label_accuracy, exact_match, rougeL
MRPC: label_accuracy, exact_match, rougeL
QQP: label_accuracy, exact_match, rougeL
SIQA: label_accuracy, exact_match, rougeL
SST2: label_accuracy, exact_match, rougeL
```

生成任务：

```text
alpaca: rougeL, exact_match
```

遗忘汇总优先使用：

```text
label_accuracy > rougeL > exact_match
```

也就是说分类任务默认用 `label_accuracy` 分析遗忘，Alpaca 默认用 `rougeL` 分析遗忘。

## 常用配置

减少评测样本数，加快调试：

```bash
MAX_EVAL_SAMPLES=100 bash run_continual_curlora.sh
```

调整每个任务的训练样本上限：

```bash
MAX_TRAIN_SAMPLES=3000 bash run_continual_curlora.sh
```

按比例抽取训练集，例如每个任务最多使用 50% 数据，同时仍受 `MAX_TRAIN_SAMPLES` 限制：

```bash
TRAIN_SAMPLE_RATIO=0.5 bash run_continual_curlora.sh
```

按比例抽取评测集：

```bash
EVAL_SAMPLE_RATIO=0.5 MAX_EVAL_SAMPLES=1000 bash run_continual_curlora.sh
```

如果要关闭分类任务分层抽样，需要手动运行 `continual_train.py` 或 `continual_eval.py` 并加入：

```bash
--no_stratified_sampling
```

增大 CUR rank：

```bash
RANK_C=32 RANK_R=32 ALPHA=64 bash run_continual_curlora.sh
```

只用 1 张卡快速 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MAX_EVAL_SAMPLES=20 EPOCHS=0.01 bash run_continual_curlora.sh
```

修改任务顺序：

```bash
vim run_continual_curlora.sh
```

修改这一行：

```bash
TASK_ORDER=(alpaca BoolQ MNLI MRPC QQP SIQA SST2)
```

## 单阶段手动训练

训练第 0 阶段 Alpaca：

```bash
torchrun --nproc_per_node 8 continual_train.py \
  --task_name alpaca \
  --stage_index 0 \
  --model_path /Users/bytedance/essential_of_lora/model/models/Qwen3-4B \
  --data_root /Users/bytedance/essential_of_lora/process_data/process_data \
  --output_root /Users/bytedance/essential_of_lora/my_curlora/continual_runs/qwen3_4b_curlora \
  --rank_c 16 \
  --rank_r 16 \
  --alpha 32 \
  --bf16 \
  --gradient_checkpointing
```

训练第 1 阶段 BoolQ，并接着第 0 阶段 adapter：

```bash
torchrun --nproc_per_node 8 continual_train.py \
  --task_name BoolQ \
  --stage_index 1 \
  --init_adapter_path continual_runs/qwen3_4b_curlora/00_alpaca/curlora_adapter.bin \
  --model_path /Users/bytedance/essential_of_lora/model/models/Qwen3-4B \
  --data_root /Users/bytedance/essential_of_lora/process_data/process_data \
  --output_root /Users/bytedance/essential_of_lora/my_curlora/continual_runs/qwen3_4b_curlora \
  --rank_c 16 \
  --rank_r 16 \
  --alpha 32 \
  --bf16 \
  --gradient_checkpointing
```

## 单阶段手动评测

评测第 1 阶段 adapter 在 Alpaca 和 BoolQ 上的表现：

```bash
python continual_eval.py \
  --adapter_path continual_runs/qwen3_4b_curlora/01_BoolQ/curlora_adapter.bin \
  --stage_name 01_BoolQ \
  --tasks alpaca BoolQ \
  --model_path /Users/bytedance/essential_of_lora/model/models/Qwen3-4B \
  --data_root /Users/bytedance/essential_of_lora/process_data/process_data \
  --output_root /Users/bytedance/essential_of_lora/my_curlora/continual_runs/qwen3_4b_curlora \
  --rank_c 16 \
  --rank_r 16 \
  --alpha 32 \
  --max_eval_samples 1000 \
  --bf16
```

## 注意事项

1. 持续训练必须保持 `rank_c`、`rank_r`、目标层和 CUR 采样配置一致，否则上一阶段 adapter 的形状可能无法加载。
2. `continual_train.py` 会先给基座模型注入 CUR-LoRA，再加载上一阶段 `curlora_adapter.bin`。
3. 训练时只更新 CUR-LoRA 的 `C/U/R` 中被设为可训练的部分，基座模型参数默认冻结。
4. 评测脚本直接加载自定义 CUR-LoRA 模块，不需要先把权重 merge 回基座模型。
5. `vLLM` 默认无法直接识别自定义 CUR-LoRA 模块；如果要用 vLLM，需要先写 merge 脚本把 `alpha * C @ U @ R` 合并进基座权重。
6. 如果评测很慢，优先调小 `MAX_EVAL_SAMPLES`，确认流程无误后再跑完整评测。

## 推荐实验记录

每次实验建议记录：

```text
任务顺序
rank_c / rank_r / alpha
是否训练 C/U/R
采样策略 normal / inverse / random
每个任务训练 epoch 或 max_steps
每阶段历史任务 label_accuracy / rougeL
forgetting_best_minus_final
drop_first_minus_final
```
