import argparse
import json
import re
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from continual_config import DEFAULT_DATA_ROOT, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_ROOT, TASKS, get_task_path, validate_tasks
from curlora_utils import (
    apply_curlora_to_model,
    build_prompt,
    load_curlora_adapter,
    load_instruction_json,
    sample_records,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CUR-LoRA adapter on continual-learning tasks.")

    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--stage_name", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--eval_sample_ratio", type=float, default=1.0)
    parser.add_argument("--no_stratified_sampling", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--rank_c", type=int, default=16)
    parser.add_argument("--rank_r", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train_C", action="store_true")
    parser.add_argument("--train_U", dest="train_U", action="store_true", default=True)
    parser.add_argument("--no_train_U", dest="train_U", action="store_false")
    parser.add_argument("--train_R", action="store_true")
    parser.add_argument("--sampling_strategy", type=str, choices=["normal", "inverse", "random"], default="normal")
    parser.add_argument("--replace", dest="replace", action="store_true", default=True)
    parser.add_argument("--no_replace", dest="replace", action="store_false")
    parser.add_argument("--adjust_dups", dest="adjust_dups", action="store_true", default=True)
    parser.add_argument("--no_adjust_dups", dest="adjust_dups", action="store_false")
    parser.add_argument("--u_init", type=str, choices=["zero", "kaiming"], default="zero")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)

    return parser.parse_args()


def normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("_", " ")
    return text


def extract_label(prediction, labels):
    norm_pred = normalize_text(prediction)
    label_map = {normalize_text(label): label for label in labels}

    for norm_label, original in label_map.items():
        if norm_pred.startswith(norm_label):
            return original
    for norm_label, original in label_map.items():
        if re.search(rf"\b{re.escape(norm_label)}\b", norm_pred):
            return original

    if set(labels) == {"A", "B", "C"}:
        match = re.search(r"\b([abc])\b", norm_pred)
        if match:
            return match.group(1).upper()
    return prediction.strip()


def make_generation_kwargs(task_name, args):
    task_cfg = TASKS[task_name]
    do_sample = args.temperature > 0
    return {
        "max_new_tokens": task_cfg["max_new_tokens"],
        "do_sample": do_sample,
        "temperature": args.temperature if do_sample else None,
        "top_p": args.top_p if do_sample else None,
        "pad_token_id": None,
    }


def batch_generate(model, tokenizer, prompts, generation_kwargs):
    generation_kwargs = dict(generation_kwargs)
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def evaluate_task(model, tokenizer, task_name, eval_path, args):
    data = load_instruction_json(eval_path)
    task_cfg = TASKS[task_name]
    labels = task_cfg["labels"]
    max_eval_samples = args.max_eval_samples
    if max_eval_samples <= 0:
        max_eval_samples = task_cfg.get("eval_sample_cap", -1)
    data, sampling_report = sample_records(
        data,
        max_samples=max_eval_samples,
        sample_ratio=args.eval_sample_ratio,
        labels=labels,
        seed=args.seed + list(TASKS).index(task_name),
        stratified=not args.no_stratified_sampling,
    )
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    predictions = []
    exact_hits = 0
    label_hits = 0
    rouge_scores = []

    generation_kwargs = make_generation_kwargs(task_name, args)

    for start in tqdm(range(0, len(data), args.batch_size), desc=f"eval:{task_name}"):
        batch = data[start : start + args.batch_size]
        prompts = [build_prompt(item["instruction"], item.get("input", "")) for item in batch]
        outputs = batch_generate(model, tokenizer, prompts, generation_kwargs)

        for item, pred in zip(batch, outputs):
            gold = item.get("output", "")
            pred_clean = pred.strip()
            gold_norm = normalize_text(gold)
            pred_norm = normalize_text(pred_clean)
            exact = int(pred_norm == gold_norm or pred_norm.startswith(gold_norm))
            exact_hits += exact

            pred_label = None
            gold_label = None
            label_correct = None
            if labels:
                pred_label = extract_label(pred_clean, labels)
                gold_label = extract_label(gold, labels)
                label_correct = int(normalize_text(pred_label) == normalize_text(gold_label))
                label_hits += label_correct

            rouge_l = scorer.score(gold, pred_clean)["rougeL"].fmeasure
            rouge_scores.append(rouge_l)

            predictions.append(
                {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "gold": gold,
                    "prediction": pred_clean,
                    "pred_label": pred_label,
                    "gold_label": gold_label,
                    "exact_match": exact,
                    "label_correct": label_correct,
                    "rougeL": rouge_l,
                }
            )

    total = max(len(data), 1)
    metrics = {
        "task": task_name,
        "num_samples": len(data),
        "sampling": sampling_report,
        "exact_match": exact_hits / total,
        "rougeL": sum(rouge_scores) / total,
    }
    if labels:
        metrics["label_accuracy"] = label_hits / total

    return metrics, predictions


def main():
    args = parse_args()
    validate_tasks(args.tasks)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = apply_curlora_to_model(model, args)
    load_curlora_adapter(model, args.adapter_path)
    model.eval()

    eval_dir = Path(args.output_root) / args.stage_name / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for task_name in args.tasks:
        eval_path = get_task_path(args.data_root, task_name, "eval")
        metrics, predictions = evaluate_task(model, tokenizer, task_name, eval_path, args)
        all_metrics.append(metrics)

        with open(eval_dir / f"{task_name}_predictions.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        with open(eval_dir / f"{task_name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    summary_path = Path(args.output_root) / "continual_metrics.jsonl"
    with open(summary_path, "a", encoding="utf-8") as f:
        for metrics in all_metrics:
            row = {"stage": args.stage_name, **metrics}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved eval metrics to {eval_dir}")


if __name__ == "__main__":
    main()
