import json
import random
import argparse
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

# --- 配置区 ---
# 基础模型路径
BASE_MODEL_PATH = "/root/shared-nvme/models/llama3_2-3b/LLM-Research/Llama-3.2-3B"
# 合并后的模型路径
MERGED_MODEL_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/curlora_llama3_2_math_merge"
# Adapter 路径
ADAPTER_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/curlora_llama3_2_math_adapter"
# 完整数据集
FULL_DATASET_PATH = "/root/shared-nvme/essence_of_lora/Dataset/meta_math_qa_finetune.json"
# 训练用过的数据集
TRAIN_DATASET_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/curlora_llama_3_2_math.json"
# 抽样数量
SAMPLE_SIZE = 1000

def prepare_test_data():
    """
    准备测试数据：从全量数据中排除训练数据，然后随机抽样。
    确保抽样结果是确定性的。
    """
    print("正在准备测试数据...")
    with open(FULL_DATASET_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 使用 instruction 作为唯一标识符
    train_instructions = {item['instruction'] for item in train_data}
    
    # 筛选出未在训练集中出现的数据
    test_pool = [item for item in full_data if item['instruction'] not in train_instructions]
    
    # --- 确保结果可复现的关键步骤 ---
    # 1. 对候选池进行排序，消除读取顺序的影响
    test_pool.sort(key=lambda x: x['instruction'])
    # 2. 设置固定随机种子
    random.seed(42)
    # --------------------------------

    print(f"全量数据: {len(full_data)} 条")
    print(f"训练数据: {len(train_instructions)} 条")
    print(f"可用于测试的数据: {len(test_pool)} 条")

    if len(test_pool) < SAMPLE_SIZE:
        print(f"⚠️ 警告: 可用测试数据不足 {SAMPLE_SIZE} 条，将使用所有可用数据。")
        return test_pool
    
    return random.sample(test_pool, SAMPLE_SIZE)

def main(mode):
    # 1. 准备数据
    test_data = prepare_test_data()
    prompts = [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        for item in test_data
    ]

    # 2. 加载 vLLM 模型
    print(f"正在加载 vLLM 模型，模式: {mode}")
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[128001, 128009] # Llama-3 EOS tokens
    )

    if mode == "merged":
        llm = LLM(model=MERGED_MODEL_PATH, trust_remote_code=True, max_model_len=4096)
        lora_request = None
    elif mode == "dynamic":
        # 启用 LoRA 支持
        llm = LLM(model=BASE_MODEL_PATH, enable_lora=True, trust_remote_code=True, max_model_len=4096)
        lora_request = LoRARequest("math_lora", 1, ADAPTER_PATH)
    elif mode == "original":
        # 仅加载原始基础模型
        llm = LLM(model=BASE_MODEL_PATH, trust_remote_code=True, max_model_len=4096)
        lora_request = None
    else:
        raise ValueError("无效的模式，请选择 'merged'、'dynamic' 或 'original'")

    # OUTPUT_FILE_PATH = f"/root/shared-nvme/essence_of_lora/infer/inference_results_vllm_{mode}.json"
    OUTPUT_FILE_PATH = f"/root/shared-nvme/essence_of_lora/my_curlora/curlora_llama_3_2_math_{mode}.json"


    # 3. 开始批量推理
    print(f"开始批量推理（共 {len(prompts)} 条）...")
    start_time = time.time()
    
    if mode == "dynamic":
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        results.append({
            "query": test_data[i]['instruction'],
            "model_res": generated_text
        })

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(test_data) if test_data else 0

    # 4. 保存结果
    print(f"\n推理完成，正在保存结果到 {OUTPUT_FILE_PATH}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每条耗时: {avg_time:.2f} 秒")
    
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("✅ 全部完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.2 vLLM 推理脚本")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["merged", "dynamic", "original"],
        required=True,
        help="选择推理模式: 'merged' (合并后的模型), 'dynamic' (动态加载 LoRA) 或 'original' (原始基础模型)。"
    )
    args = parser.parse_args()
    main(args.mode)

