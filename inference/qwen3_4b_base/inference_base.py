import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer # 引入 tokenizer 用于构建对话模板


# 配置路径
MODEL_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/model/Qwen3-4B/Qwen/Qwen3-4B"
DATASET_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/process_data/MNLI/mnli_processed/test_new_mnil.json"
OUTPUT_PATH = "/root/shared-nvme/essence_of_lora/my_curlora/inference/qwen3_4b_base/mnli/test_mnli_results.json"

def main():
    # 1. 加载数据集
    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


    # 2. 准备 prompt
    prompts = []
    for item in dataset:
        # instruction = item.get("instruction", "")
        # input_text = item.get("input", "")
        # # 直接拼接内容，不包含 "Instruction:", "Input:", "Response:" 等标签
        # if input_text:
        #     prompt = f"{instruction}\n{input_text}"
        # else:
        #     prompt = f"{instruction}"
        # prompts.append(prompt)
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        
        # 构建符合 Qwen3 格式的对话内容
        content = instruction
        if input_text:
            content += f"\n{input_text}"
            
        messages = [
            {"role": "user", "content": content}
        ]
        
        # 使用 apply_chat_template 自动生成标准 Prompt (包含 <|im_start|> 等标签)
        # add_generation_prompt=True 确保结尾是 assistant 的开始标记，引导模型生成
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # 如果支持
        )
        prompts.append(prompt_text)

    # 3. 初始化 vLLM
    print(f"Initializing vLLM with base model: {MODEL_PATH}...")
    try:
        # 针对 Qwen3-4B 设置合适的参数
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="bfloat16",
            enforce_eager=True
        )
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return

    # 4. 设置采样参数，添加惩罚项以缓解复读机现象
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1, # 重复惩罚，大于1.0的值会抑制重复
        presence_penalty=0.1,   # 存在惩罚，鼓励模型谈论新话题
        frequency_penalty=0.1,  # 频率惩罚，降低已出现词的生成概率
        max_tokens=512,
        stop=["<|endoftext|>", "<|im_end|>"] # Qwen 3 正确的停止符
    )

    # 5. 执行推理
    print(f"Starting inference on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. 保存结果
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        results.append({
            "instruction": dataset[i].get("instruction", ""),
            "input": dataset[i].get("input", ""),
            "output": generated_text
        })

    print(f"Saving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
