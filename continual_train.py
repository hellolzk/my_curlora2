import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from continual_config import DEFAULT_DATA_ROOT, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_ROOT, TASKS, get_task_path
from curlora_utils import (
    apply_curlora_to_model,
    is_main_process,
    load_curlora_adapter,
    make_data_collator,
    prepare_sft_dataset,
    save_curlora_adapter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train one continual CUR-LoRA stage.")

    parser.add_argument("--task_name", type=str, required=True, choices=list(TASKS))
    parser.add_argument("--stage_index", type=int, required=True)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--init_adapter_path", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")

    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--rank_c", type=int, default=16)
    parser.add_argument("--rank_r", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
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

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--train_sample_ratio", type=float, default=1.0)
    parser.add_argument("--no_stratified_sampling", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--optim", type=str, default="adamw_torch")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stage_name = f"{args.stage_index:02d}_{args.task_name}"
    output_dir = Path(args.output_root) / stage_name
    train_path = get_task_path(args.data_root, args.task_name, "train")
    task_cfg = TASKS[args.task_name]
    max_train_samples = args.max_train_samples
    if max_train_samples <= 0:
        max_train_samples = task_cfg.get("train_sample_cap", -1)

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Stage: {stage_name}")
        print(f"Train data: {train_path}")
        print(f"Max train samples: {max_train_samples}")
        print(f"Train sample ratio: {args.train_sample_ratio}")
        print(f"Stratified sampling: {not args.no_stratified_sampling and bool(task_cfg.get('labels'))}")
        print(f"Output dir: {output_dir}")
        print(f"Init adapter: {args.init_adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    model = apply_curlora_to_model(model, args)
    load_curlora_adapter(model, args.init_adapter_path)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset, sampling_report = prepare_sft_dataset(
        tokenizer,
        train_path,
        args.max_length,
        task_cfg=task_cfg,
        max_train_samples=max_train_samples,
        train_sample_ratio=args.train_sample_ratio,
        seed=args.seed + args.stage_index,
        stratified=not args.no_stratified_sampling,
    )
    if is_main_process():
        print("Sampling report:")
        print(json.dumps(sampling_report, indent=2, ensure_ascii=False))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_data_collator(tokenizer),
    )

    trainer.train()

    if trainer.is_world_process_zero():
        adapter_path = save_curlora_adapter(model, output_dir)
        config = vars(args)
        config["stage_name"] = stage_name
        config["train_path"] = train_path
        config["effective_max_train_samples"] = max_train_samples
        config["sampling_report"] = sampling_report
        config["adapter_path"] = adapter_path
        with open(output_dir / "stage_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        with open(output_dir / "adapter_path.txt", "w", encoding="utf-8") as f:
            f.write(adapter_path + "\n")
        print(f"Saved stage adapter: {adapter_path}")


if __name__ == "__main__":
    main()
