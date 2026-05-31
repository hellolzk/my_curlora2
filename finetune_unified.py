import os
import json
import argparse
import torch
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from unified_curlora import LinearWithCURLoRA, CURModule

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model with CUR-LoRA")
    
    # 模型和数据路径
    parser.add_argument("--model_path", type=str, default="/root/shared-nvme/essence_of_lora/my_curlora/model/Qwen3-4B/Qwen/Qwen3-4B")
    parser.add_argument("--dataset_path", type=str, default="/root/shared-nvme/essence_of_lora/my_curlora/process_data/alpaca/train_alpaca.json")
    parser.add_argument("--output_base_dir", type=str, default="/root/shared-nvme/essence_of_lora/my_curlora/curlora_adapter/qwen3_4b")
    
    # CUR 超参数
    parser.add_argument("--rank", type=int, default=None, help="Unified rank for both C and R")
    parser.add_argument("--rank_c", type=int, default=16, help="Rank for C matrix (columns)")
    parser.add_argument("--rank_r", type=int, default=16, help="Rank for R matrix (rows)")
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for CUR module")
    parser.add_argument("--train_C", action="store_true", help="Whether to train C matrix")
    parser.add_argument("--train_U", dest="train_U", action="store_true", default=True, help="Whether to train U matrix")
    parser.add_argument("--no_train_U", dest="train_U", action="store_false", help="Freeze U matrix and initialize it with pseudo-inverse")
    parser.add_argument("--train_R", action="store_true", help="Whether to train R matrix")
    parser.add_argument("--sampling_strategy", type=str, choices=["normal", "inverse", "random"], default="normal")
    parser.add_argument("--replace", dest="replace", action="store_true", default=True, help="Sampling with replacement")
    parser.add_argument("--no_replace", dest="replace", action="store_false", help="Sampling without replacement")
    parser.add_argument("--adjust_dups", dest="adjust_dups", action="store_true", default=True, help="Adjust duplicates in sampling")
    parser.add_argument("--no_adjust_dups", dest="adjust_dups", action="store_false", help="Keep duplicate samples instead of merging them")
    parser.add_argument("--u_init", type=str, choices=["zero", "kaiming"], default="zero")
    
    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=2e-4)#学习率
    parser.add_argument("--num_train_epochs", type=int, default=3)#epochs
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)#每个gpu同时处理的样本数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)#累积多少次梯度才更新一次参数
    parser.add_argument("--logging_steps", type=int, default=10)#每10步打印一次loss到控制台
    parser.add_argument("--save_steps", type=int, default=100)#每100步保存一次checkpoint
    parser.add_argument("--max_length", type=int, default=512)#输入+输出的最大token数（超出截断）
    parser.add_argument("--warmup_ratio", type=float, default=0.1)#前10%的训练步数线性增加学习率到设定值
    parser.add_argument("--weight_decay", type=float, default=0.01)#L2正则化系数，防止过拟合
    parser.add_argument("--bf16", action="store_true", default=True)#使用bfloat16混合精度训练
    parser.add_argument("--lr_scheduler_type", type=str, choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], default="cosine")#学习率调度器
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")#最大步数
    parser.add_argument("--seed", type=int, default=42)#随机种子
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")#梯度检查点
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer to use")#优化器
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints. Deletes the older checkpoints.")#检查点数量限制
    
    return parser.parse_args()

def apply_curlora_to_model(model, args):
    print(f"Applying CUR-LoRA with rank={args.rank}, alpha={args.alpha}...")
    
    # 针对 Qwen2/Qwen3 的典型层
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 冻结原始模型参数
    for param in model.parameters():
        param.requires_grad = False
        
    modules_to_replace = []
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, torch.nn.Linear):
            modules_to_replace.append((name, module))
            
    for module_index, (name, module) in enumerate(modules_to_replace):
        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
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
            seed=args.seed + module_index
        )
        setattr(parent, child_name, new_module)
        
    print(f"Applied CUR-LoRA to {len(modules_to_replace)} layers.")
    return model

def prepare_dataset(tokenizer, dataset_path, max_length):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    def tokenize_function(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        
        prompts = []
        for i, inp, out in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction: {i}\nInput: {inp}\nResponse: "
            else:
                prompt = f"Instruction: {i}\nResponse: "
            prompts.append(prompt + out + tokenizer.eos_token)
            
        model_inputs = tokenizer(prompts, truncation=True, padding=False, max_length=max_length)
        
        labels = []
        for i in range(len(prompts)):
            prompt_only = prompts[i].split(outputs[i])[0]
            tokenized_prompt = tokenizer(prompt_only, truncation=True, max_length=max_length)
            prompt_len = len(tokenized_prompt["input_ids"])
            
            label = [-100] * prompt_len + model_inputs["input_ids"][i][prompt_len:]
            labels.append(label)
            
        model_inputs["labels"] = labels
        return model_inputs

    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def save_curlora_adapter(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    adapter_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, CURModule):
            prefix = name + "."
            # 始终保存 C, U, R，无论是否可训练，因为推理需要
            adapter_state_dict[prefix + "C"] = module.C.data.cpu()
            adapter_state_dict[prefix + "U"] = module.U.data.cpu()
            adapter_state_dict[prefix + "R"] = module.R.data.cpu()
            
    torch.save(adapter_state_dict, os.path.join(output_dir, "curlora_adapter.bin"))
    print(f"Adapter saved to {output_dir}")

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 自动生成包含数据集名和时间戳的输出目录
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(args.output_base_dir, f"{dataset_name}_{timestamp}")
    
    print(f"Loading model and tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # 应用 CUR-LoRA
    model = apply_curlora_to_model(model, args)
    
    # 针对梯度检查点 (Gradient Checkpointing) 的关键修复：
    # 当原始模型被冻结且开启了梯度检查点时，必须显式启用输入层的梯度计算，
    # 否则 PyTorch 无法构建反向传播计算图，导致 "does not require grad" 错误。
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")
    
    # 准备数据
    print("Preparing dataset...")
    train_dataset = prepare_dataset(tokenizer, args.dataset_path, args.max_length)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join(final_output_dir, "checkpoints"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=args.bf16,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        save_total_limit=args.save_total_limit,
        report_to="none"
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    
    print("Starting training...")
    trainer.train()
    
    # 保存 adapter
    save_curlora_adapter(model, final_output_dir)
    
    # 保存配置文件记录超参数
    config = vars(args)
    config["timestamp"] = timestamp
    config["final_output_dir"] = final_output_dir
    with open(os.path.join(final_output_dir, "finetune_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    print("Done!")

if __name__ == "__main__":
    main()
