import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq

from unified_curlora import CURModule, LinearWithCURLoRA


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def build_prompt(instruction, input_text):
    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}\nResponse: "
    return f"Instruction: {instruction}\nResponse: "


def load_instruction_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return data


def normalize_label_text(text):
    text = str(text).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", text)


def infer_record_label(record, labels):
    if not labels:
        return None

    output = str(record.get("output", "")).strip()
    normalized_output = normalize_label_text(output)

    if set(labels) == {"A", "B", "C"}:
        match = re.match(r"^\s*([ABCabc])\b", output)
        if match:
            return match.group(1).upper()

    for label in labels:
        normalized_label = normalize_label_text(label)
        if normalized_output == normalized_label or normalized_output.startswith(normalized_label):
            return label

    return output


def _largest_remainder_alloc(group_sizes, sample_size):
    total = sum(group_sizes.values())
    if sample_size >= total:
        return dict(group_sizes)

    raw_alloc = {
        label: sample_size * size / total
        for label, size in group_sizes.items()
    }
    alloc = {
        label: min(size, int(raw_alloc[label]))
        for label, size in group_sizes.items()
    }

    for label, size in group_sizes.items():
        if size > 0 and alloc[label] == 0:
            alloc[label] = 1

    while sum(alloc.values()) > sample_size:
        candidates = [label for label in alloc if alloc[label] > 0]
        label = min(candidates, key=lambda item: raw_alloc[item] - int(raw_alloc[item]))
        alloc[label] -= 1

    while sum(alloc.values()) < sample_size:
        candidates = [label for label, size in group_sizes.items() if alloc[label] < size]
        if not candidates:
            break
        label = max(candidates, key=lambda item: raw_alloc[item] - int(raw_alloc[item]))
        alloc[label] += 1

    return alloc


def sample_records(records, max_samples=-1, sample_ratio=1.0, labels=None,
                   seed=42, stratified=True):
    total = len(records)
    if total == 0:
        return [], {"original_count": 0, "sampled_count": 0, "label_counts": {}}
    if sample_ratio <= 0:
        raise ValueError(f"sample_ratio must be > 0, got {sample_ratio}")

    ratio_count = int(total * sample_ratio) if sample_ratio < 1.0 else total
    cap_count = total if max_samples is None or max_samples <= 0 else min(max_samples, total)
    sample_size = max(1, min(total, ratio_count, cap_count))

    rng = random.Random(seed)
    original_label_counts = Counter(infer_record_label(item, labels) for item in records) if labels else Counter()

    if sample_size >= total:
        sampled = list(records)
        rng.shuffle(sampled)
    elif labels and stratified:
        groups = defaultdict(list)
        for item in records:
            groups[infer_record_label(item, labels)].append(item)

        group_sizes = {label: len(items) for label, items in groups.items()}
        alloc = _largest_remainder_alloc(group_sizes, sample_size)

        sampled = []
        for label, items in groups.items():
            items = list(items)
            rng.shuffle(items)
            sampled.extend(items[:alloc.get(label, 0)])
        rng.shuffle(sampled)
    else:
        sampled = rng.sample(records, sample_size)

    sampled_label_counts = Counter(infer_record_label(item, labels) for item in sampled) if labels else Counter()
    return sampled, {
        "original_count": total,
        "sampled_count": len(sampled),
        "sample_ratio": sample_ratio,
        "max_samples": max_samples,
        "stratified": bool(labels and stratified),
        "label_counts": dict(original_label_counts),
        "sampled_label_counts": dict(sampled_label_counts),
    }


def prepare_sft_dataset(tokenizer, dataset_path, max_length, task_cfg=None,
                        max_train_samples=-1, train_sample_ratio=1.0,
                        seed=42, stratified=True):
    data = load_instruction_json(dataset_path)
    labels = task_cfg.get("labels") if task_cfg else None
    data, sampling_report = sample_records(
        data,
        max_samples=max_train_samples,
        sample_ratio=train_sample_ratio,
        labels=labels,
        seed=seed,
        stratified=stratified,
    )

    def tokenize_function(examples):
        prompts = [
            build_prompt(inst, inp)
            for inst, inp in zip(examples["instruction"], examples["input"])
        ]
        full_texts = [
            prompt + output + tokenizer.eos_token
            for prompt, output in zip(prompts, examples["output"])
        ]

        model_inputs = tokenizer(
            full_texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )

        labels = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
            prompt_len = min(len(prompt_ids), len(model_inputs["input_ids"][idx]))
            label = [-100] * prompt_len + model_inputs["input_ids"][idx][prompt_len:]
            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    dataset = Dataset.from_list(data)
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized, sampling_report


def make_data_collator(tokenizer):
    return DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )


def apply_curlora_to_model(model, args):
    for param in model.parameters():
        param.requires_grad = False

    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in TARGET_MODULES):
            modules_to_replace.append((name, module))

    for module_index, (name, module) in enumerate(modules_to_replace):
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

        new_module = LinearWithCURLoRA(
            module,
            rank=args.rank,
            rank_c=args.rank_c,
            rank_r=args.rank_r,
            alpha=args.alpha,
            dropout=args.dropout,
            train_C=args.train_C,
            train_U=args.train_U,
            train_R=args.train_R,
            sampling_strategy=args.sampling_strategy,
            replace=args.replace,
            adjust_dups=args.adjust_dups,
            u_init=args.u_init,
            seed=args.seed + module_index,
        )
        setattr(parent, child_name, new_module)

    print(f"Applied CUR-LoRA to {len(modules_to_replace)} linear layers.")
    return model


def find_adapter_bin(adapter_path):
    if adapter_path is None:
        return None
    path = Path(adapter_path)
    if path.is_dir():
        path = path / "curlora_adapter.bin"
    if not path.exists():
        raise FileNotFoundError(f"Adapter file not found: {path}")
    return str(path)


def load_curlora_adapter(model, adapter_path, strict_shapes=True):
    adapter_bin = find_adapter_bin(adapter_path)
    if adapter_bin is None:
        return

    state = torch.load(adapter_bin, map_location="cpu")
    missing = []
    loaded = 0

    for name, module in model.named_modules():
        if not isinstance(module, CURModule):
            continue
        prefix = name + "."
        keys = {
            "C": prefix + "C",
            "U": prefix + "U",
            "R": prefix + "R",
        }
        if not all(key in state for key in keys.values()):
            missing.append(name)
            continue

        for attr, key in keys.items():
            target = getattr(module, attr)
            value = state[key].to(device=target.device, dtype=target.dtype)
            if strict_shapes and tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
                )
            target.data.copy_(value)
        loaded += 1

    if missing:
        raise KeyError(f"Missing adapter weights for CUR modules: {missing[:5]} ... total={len(missing)}")
    print(f"Loaded CUR-LoRA adapter from {adapter_bin}; modules loaded: {loaded}")


def save_curlora_adapter(model, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adapter_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, CURModule):
            prefix = name + "."
            adapter_state_dict[prefix + "C"] = module.C.detach().cpu()
            adapter_state_dict[prefix + "U"] = module.U.detach().cpu()
            adapter_state_dict[prefix + "R"] = module.R.detach().cpu()

    torch.save(adapter_state_dict, output_path / "curlora_adapter.bin")
    return str(output_path / "curlora_adapter.bin")
