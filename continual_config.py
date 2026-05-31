from pathlib import Path


DEFAULT_MODEL_PATH = "/Users/bytedance/essential_of_lora/model/models/Qwen3-4B"
DEFAULT_DATA_ROOT = "/Users/bytedance/essential_of_lora/process_data/process_data"
DEFAULT_OUTPUT_ROOT = "/Users/bytedance/essential_of_lora/my_curlora/continual_runs/qwen3_4b_curlora"


TASK_ORDER = [
    "alpaca",
    "BoolQ",
    "MNLI",
    "MRPC",
    "QQP",
    "SIQA",
    "SST2",
]


TASKS = {
    "alpaca": {
        "train": "alpaca/train_alpaca.json",
        "eval": "alpaca/test_alpaca.json",
        "metric_type": "generation",
        "max_new_tokens": 256,
        "train_sample_cap": 5000,
        "eval_sample_cap": 1000,
        "labels": None,
    },
    "BoolQ": {
        "train": "BoolQ/boolq_processed/train.json",
        "eval": "BoolQ/boolq_processed/validation.json",
        "metric_type": "classification",
        "max_new_tokens": 8,
        "train_sample_cap": 5000,
        "eval_sample_cap": 1000,
        "labels": ["yes", "no"],
    },
    "MNLI": {
        "train": "MNLI/mnli_processed/train_new_mnli.json",
        "eval": "MNLI/mnli_processed/test_new_mnil.json",
        "metric_type": "classification",
        "max_new_tokens": 12,
        "train_sample_cap": 5000,
        "eval_sample_cap": 1000,
        "labels": ["entailment", "neutral", "contradiction"],
    },
    "MRPC": {
        "train": "MRPC/mrpc_processed/train.json",
        "eval": "MRPC/mrpc_processed/validation.json",
        "metric_type": "classification",
        "max_new_tokens": 12,
        "train_sample_cap": 5000,
        "eval_sample_cap": 1000,
        "labels": ["equivalent", "not equivalent"],
    },
    "QQP": {
        "train": "QQP/qqp_processed/train.json",
        "eval": "QQP/qqp_processed/validation.json",
        "metric_type": "classification",
        "max_new_tokens": 12,
        "train_sample_cap": 5000,
        "eval_sample_cap": 1000,
        "labels": ["duplicate", "not_duplicate"],
    },
    "SIQA": {
        "train": "SIQA/siqa_processed/train.json",
        "eval": "SIQA/siqa_processed/validation.json",
        "metric_type": "classification",
        "max_new_tokens": 48,
        "train_sample_cap": 5000,
        "eval_sample_cap": 1000,
        "labels": ["A", "B", "C"],
    },
    "SST2": {
        "train": "SST2/sst2_processed/train.json",
        "eval": "SST2/sst2_processed/validation.json",
        "metric_type": "classification",
        "max_new_tokens": 8,
        "train_sample_cap": 5000,
        "eval_sample_cap": 1000,
        "labels": ["positive", "negative"],
    },
}


def get_task_path(data_root, task_name, split):
    task_cfg = TASKS[task_name]
    return str(Path(data_root) / task_cfg[split])


def validate_tasks(task_names):
    unknown = [name for name in task_names if name not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid tasks: {list(TASKS)}")
