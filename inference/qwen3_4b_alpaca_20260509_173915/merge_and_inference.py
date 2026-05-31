import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import gc

# --- 配置路径 ---
BASE_MODEL_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/model/Qwen3-4B/Qwen/Qwen3-4B"
ADAPTER_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/curlora_adapter/qwen3_4b/train_alpaca_20260509_173915/curlora_adapter.bin"
DATASET_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/process_data/alpaca/test_alpaca.json"
INFERENCE_DIR = "/root/shared-nvme/essence_of_lora/my_curlora/inference/qwen3_4b_alpaca_20260509_173915"
TEMP_MERGED_MODEL_DIR = os.path.join(INFERENCE_DIR, "temp_merged_model")
OUTPUT_JSON_PATH = os.path.join(INFERENCE_DIR, "test_alpaca_results.json")

def merge_weights():
    """
    加载基础模型和 CUR 适配器，执行合并，并保存到临时目录。
    vLLM 无法直接加载自定义的 CUR 结构，因此必须先合并为标准权重。
    """
    print("🚀 正在加载基础模型以进行权重合并...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # 在 CPU 上合并以节省显存
        trust_remote_code=True
    )

    print(f"📂 正在加载适配器: {ADAPTER_PATH}")
    adapter_state_dict = torch.load(ADAPTER_PATH, map_location="cpu")

    # 获取微调配置（用于获取 alpha）
    config_path = os.path.join(os.path.dirname(ADAPTER_PATH), "finetune_config.json")
    alpha = 32 # 默认值
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            alpha = config.get("alpha", 32)
    
    print(f"🔄 正在合并权重 (Alpha={alpha})...")
    # 遍历 state_dict 中的所有 CUR 参数
    # 格式通常是: model.layers.0.self_attn.q_proj.curlora.U
    
    # 整理出每个层的参数
    layer_params = {}
    for key, value in adapter_state_dict.items():
        if ".curlora." in key:
            base_key = key.split(".curlora.")[0]
            param_name = key.split(".curlora.")[1] # C, U, R
            if base_key not in layer_params:
                layer_params[base_key] = {}
            layer_params[base_key][param_name] = value

    for base_key, params in layer_params.items():
        if "C" in params and "U" in params and "R" in params:
            # 计算增量权重 Delta_W = alpha * (C @ U @ R)
            C, U, R = params["C"], params["U"], params["R"]
            delta_W = alpha * torch.matmul(torch.matmul(C, U), R)
            
            # 找到原始模型的权重参数名
            # base_key 可能是 "model.layers.0.self_attn.q_proj"
            weight_key = base_key + ".weight"
            
            if weight_key in model.state_dict():
                # 合并：W_new = W_old + delta_W
                # 注意：确保 dtype 一致
                model.state_dict()[weight_key].data += delta_W.to(model.dtype)
            else:
                print(f"⚠️ 警告: 找不到权重键 {weight_key}")

    print(f"💾 正在保存合并后的模型到临时目录: {TEMP_MERGED_MODEL_DIR}")
    model.save_pretrained(TEMP_MERGED_MODEL_DIR)
    tokenizer.save_pretrained(TEMP_MERGED_MODEL_DIR)
    
    # 释放显存/内存
    del model
    gc.collect()

def run_vllm_inference():
    """
    使用 vLLM 加载合并后的模型并进行推理。
    """
    print("📡 正在启动 vLLM 推理引擎...")
    
    # 1. 初始化 vLLM
    # enforce_eager=True 可以跳过 CUDA Graph 的捕获阶段，显著加快启动速度
    llm = LLM(
        model=TEMP_MERGED_MODEL_DIR,
        trust_remote_code=True,
        gpu_memory_utilization=0.9, # 统一为 0.9
        max_model_len=4096,
        dtype="bfloat16",
        enforce_eager=True 
    )

    tokenizer = llm.get_tokenizer()

    # 2. 加载数据集并准备 prompt
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"📝 正在使用 Chat Template 格式化 {len(dataset)} 条 Prompt...")
    prompts = []
    for item in dataset:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        
        # 构建符合 Qwen3 格式的对话内容
        content = instruction
        if input_text:
            content += f"\n{input_text}"
            
        messages = [
            {"role": "user", "content": content}
        ]
        
        # 使用模型的 chat_template 进行格式化，并添加生成引导符
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # 如果支持
        )
        prompts.append(prompt)

    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1, 
        presence_penalty=0.1,   
        frequency_penalty=0.1,  
        max_tokens=512,
        stop=["<|endoftext|>", "<|im_end|>"] 
    )

    print(f"✍️ 正在执行批量推理...")
    outputs = llm.generate(prompts, sampling_params)

    # 整理结果
    final_results = []
    for i, output in enumerate(outputs):
        final_results.append({
            "instruction": dataset[i].get("instruction", ""),
            "input": dataset[i].get("input", ""),
            "output": output.outputs[0].text
        })

    # 保存 JSON
    print(f"📊 正在保存推理结果到: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print("✅ 推理完成！")

if __name__ == "__main__":
    # 1. 合并模型
    if not os.path.exists(os.path.join(TEMP_MERGED_MODEL_DIR, "config.json")):
        merge_weights()
    else:
        print("ℹ️ 检测到已存在的合并模型，跳过合并步骤。")
    
    # 2. 推理
    run_vllm_inference()
