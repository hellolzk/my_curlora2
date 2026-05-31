import json
import numpy as np
from bert_score import score

def load_json_outputs(file_path):
    """从 JSON 文件中提取 output 字段"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [item.get("output", "") for item in data]

def calculate_bertscore(file1_path, file2_path):
    print(f"正在读取参考文件: {file1_path}", flush=True)
    references = load_json_outputs(file1_path)

    print(f"正在读取生成文件: {file2_path}", flush=True)
    candidates = load_json_outputs(file2_path)

    # 长度对齐
    if len(references) != len(candidates):
        print(
            f"警告：两个文件条数不一致 "
            f"(ref={len(references)}, cand={len(candidates)})",
            flush=True
        )

        min_len = min(len(references), len(candidates))
        references = references[:min_len]
        candidates = candidates[:min_len]

    print(f"开始计算 BERTScore，共 {len(references)} 条数据...", flush=True)

    # =========================
    # BERTScore 核心计算
    # =========================
    P, R, F1 = score(
        candidates,          # 模型输出
        references,          # 参考答案
        lang="en",           # 英文任务
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=True,
        device="cuda",        # 如果没有 GPU 改成 "cpu"
        batch_size=8
    )

    # 转 numpy
    precision = P.mean().item()
    recall = R.mean().item()
    f1 = F1.mean().item()

    # =========================
    # 输出结果
    # =========================
    print("\n" + "=" * 50)
    print("📊 BERTScore 评估结果")
    print("=" * 50)

    print(f"BERTScore Precision : {precision:.4f}")
    print(f"BERTScore Recall    : {recall:.4f}")
    print(f"BERTScore F1        : {f1:.4f}")

    print("=" * 50)

if __name__ == "__main__":

    # 参考答案
    file1 = "/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2process_data/alpaca/test_alpaca.json"

    # 模型生成结果
    file2 = "/mnt/bn/chenhaobo-va-data/liuzekun2/my_curlora2/inference/qwen3_4b_alpaca_20260509_173915/test_alpaca_results.json"

    calculate_bertscore(file1, file2)