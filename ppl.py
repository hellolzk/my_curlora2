import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


# 配置
MODEL_PATH = "/mnt/bn/chenhaobo-va-data/liuzekun2/models/Qwen3-4B"
DATASET_PATH = "/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2/dataset/wikitext-2-raw-v1/test/test-00000-of-00001.parquet"
MAX_LENGTH = 1024  # 根据显存调整，Qwen2.5-7B 在 A100/A800 上可以跑更大，但 1024 比较稳妥

def calculate_ppl_qwen2_5():
    # 1. 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Qwen2 默认没有 pad_token，需要设置
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 2. 加载数据集
    print("加载数据集...")
    # 直接加载本地 parquet 文件
    data_files = {"test": DATASET_PATH}
    raw_datasets = load_dataset("parquet", data_files=data_files)
    test_data = raw_datasets["test"]

    # 3. 定义处理逻辑
    # 我们不在这里做 batch tokenization，而是逐行处理以便更灵活地计算 loss
    # 并跳过标题行
    
    total_loss = 0
    total_tokens = 0
    
    # 使用 tqdm 显示进度条
    for row in tqdm(test_data, desc="计算困惑度"):
        text = row['text']
        
        # --- 数据清洗逻辑 (基于截图) ---
        # 1. 去除首尾空白
        text = text.strip()
        # 2. 跳过空行
        if len(text) == 0:
            continue
        # 3. 跳过 Wiki 标题行 (以 = 开头)
        # 截图显示: "= Robert Boulter =", "= = Career = ="
        if text.startswith("="):
            continue
            
        # --- Tokenization ---
        # Qwen2 需要显式处理输入
        inputs = tokenizer(text, return_tensors="pt", truncation=False) # 先不截断，看多长
        
        input_ids = inputs.input_ids.to(model.device)
        seq_len = input_ids.size(1)
        
        # 如果文本太长，需要分块处理 (Sliding Window 或 Independent Chunks)
        # 这里使用 Independent Chunks 逻辑
        num_chunks = (seq_len + MAX_LENGTH - 1) // MAX_LENGTH
        
        with torch.no_grad():
            for i in range(num_chunks):
                start = i * MAX_LENGTH
                end = min((i + 1) * MAX_LENGTH, seq_len)
                
                # 获取当前块
                chunk_ids = input_ids[:, start:end]
                
                # 如果块太小（例如只有1个token），无法计算 loss (需要 input 和 label)
                if chunk_ids.size(1) < 2:
                    continue
                
                # 模型推理
                # labels 就是 inputs，模型内部会计算 shift
                outputs = model(chunk_ids, labels=chunk_ids)
                loss = outputs.loss
                
                # 累加 Loss * Token数量 (因为 loss 是平均值)
                # loss.item() 是该 chunk 的平均 loss
                total_loss += loss.item() * chunk_ids.size(1)
                total_tokens += chunk_ids.size(1)

    # 4. 计算最终困惑度
    if total_tokens == 0:
        print("没有有效的 tokens 被计算。")
        return
        
    avg_nll = total_loss / total_tokens
    ppl = np.exp(avg_nll)
    
    print("-" * 30)
    print(f"有效 Token 数: {total_tokens}")
    print(f"平均负对数似然 (NLL): {avg_nll:.4f}")
    print(f"困惑度 (Perplexity): {ppl:.4f}")
    print("-" * 30)


def calculate_ppl_glm4():
    # 1. 加载模型和分词器
    print("加载 GLM-4 模型...")
    
    # GLM-4 分词器通常自带 pad_token，不需要强制指定为 eos_token
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True,
        padding_side="left" # 计算 PPL 时，padding 侧通常设为 left
    )
    
    # 兼容性检查：如果确实没有 pad_token 再设置，且不要用 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token # GLM-4 通常使用 unk_token 作为 pad

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, # GLM-4 推荐使用 bfloat16
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 2. 加载数据集
    print("加载数据集...")
    data_files = {"test": DATASET_PATH}
    raw_datasets = load_dataset("parquet", data_files=data_files)
    test_data = raw_datasets["test"]

    total_loss = 0.0
    total_tokens = 0

    # 3. 循环计算
    for row in tqdm(test_data, desc="计算 PPL"):
        text = row['text']
        
        # --- 数据清洗 ---
        text = text.strip()
        if len(text) == 0 or text.startswith("="):
            continue
            
        # --- Tokenization ---
        # 注意：GLM-4 对长文本支持较好，这里尝试不截断，让模型自己处理
        # 如果显存爆了，再考虑开启 truncation
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=False, 
            max_length=MAX_LENGTH 
        )
        
        input_ids = inputs.input_ids.to(model.device)
        seq_len = input_ids.size(1)
        
        # 如果单条文本超过 MAX_LENGTH，使用滑动窗口或分块
        # 这里保留你的分块逻辑，但要注意 GLM-4 的上下文连贯性
        if seq_len > MAX_LENGTH:
             # 简单的分块策略（会丢失块与块之间的关联，导致 PPL 略微偏高）
             num_chunks = (seq_len + MAX_LENGTH - 1) // MAX_LENGTH
        else:
             num_chunks = 1

        with torch.no_grad():
            for i in range(num_chunks):
                # 计算当前块的起止位置
                start = i * MAX_LENGTH
                end = min((i + 1) * MAX_LENGTH, seq_len)
                
                # 提取当前块
                chunk_ids = input_ids[:, start:end]
                
                if chunk_ids.size(1) < 2: continue

                # 计算 Loss
                # GLM-4 是 CausalLM，labels 等于 input_ids 即可
                outputs = model(chunk_ids, labels=chunk_ids)
                loss = outputs.loss
                
                # 累加
                # 注意：loss 是平均值，需要乘以 token 数还原总 loss
                total_loss += loss.item() * chunk_ids.size(1)
                total_tokens += chunk_ids.size(1)

    # 4. 输出结果
    if total_tokens == 0:
        print("未计算任何 Token")
        return

    avg_nll = total_loss / total_tokens
    ppl = np.exp(avg_nll)
    
    print("-" * 30)
    print(f"模型: {MODEL_PATH}")
    print(f"总 Token 数: {total_tokens}")
    print(f"平均负对数似然 (NLL): {avg_nll:.4f}")
    print(f"困惑度 (Perplexity): {ppl:.4f}")
    print("-" * 30)


def calculate_ppl_llama3_2():
    # 1. 加载模型和分词器
    print(f"加载 Llama 3.2 模型: {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Llama 3.2 分词器可能没有 pad_token，需要手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, # Llama 3.2 推荐使用 bfloat16
        device_map="auto",          # 自动将模型分配到可用设备 (CPU/GPU)
    )
    model.eval()

    # 2. 加载数据集
    print("加载数据集...")
    data_files = {"test": DATASET_PATH}
    raw_datasets = load_dataset("parquet", data_files=data_files)
    test_data = raw_datasets["test"]

    total_loss = 0.0
    total_tokens = 0

    # 3. 循环计算
    for row in tqdm(test_data, desc="计算 PPL"):
        text = row['text']
        
        # --- 数据清洗 ---
        text = text.strip()
        if len(text) == 0 or text.startswith("="):
            continue
            
        # --- Tokenization ---
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=False, 
            max_length=MAX_LENGTH 
        )
        
        input_ids = inputs.input_ids.to(model.device)
        seq_len = input_ids.size(1)
        
        # 如果单条文本超过 MAX_LENGTH，使用分块策略
        if seq_len > MAX_LENGTH:
             num_chunks = (seq_len + MAX_LENGTH - 1) // MAX_LENGTH
        else:
             num_chunks = 1

        with torch.no_grad():
            for i in range(num_chunks):
                # 计算当前块的起止位置
                start = i * MAX_LENGTH
                end = min((i + 1) * MAX_LENGTH, seq_len)
                
                # 提取当前块
                chunk_ids = input_ids[:, start:end]
                
                if chunk_ids.size(1) < 2: continue

                # 计算 Loss
                # Llama 是 CausalLM，labels 等于 input_ids 即可
                outputs = model(chunk_ids, labels=chunk_ids)
                loss = outputs.loss
                
                # 累加
                # loss 是平均值，需要乘以 token 数还原总 loss
                total_loss += loss.item() * chunk_ids.size(1)
                total_tokens += chunk_ids.size(1)

    # 4. 输出结果
    if total_tokens == 0:
        print("未计算任何 Token")
        return

    avg_nll = total_loss / total_tokens
    ppl = np.exp(avg_nll)
    
    print("-" * 30)
    print(f"模型: {MODEL_PATH}")
    print(f"总 Token 数: {total_tokens}")
    print(f"平均负对数似然 (NLL): {avg_nll:.4f}")
    print(f"困惑度 (Perplexity): {ppl:.4f}")
    print("-" * 30)
if __name__ == "__main__":
    calculate_ppl_qwen2_5()