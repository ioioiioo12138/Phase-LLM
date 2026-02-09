import argparse
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

# === 配置区域 ===
LABELS = [
    "L12 phase does not form",
    "Only FCC and L12 phases form",
    "Other phases form in addition to FCC and L12"
]

DEFAULT_SYSTEM_PROMPT = "You are a senior metallurgist. Answer the question after a comprehensive reasoning process."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/user_liang/materials/LlamaFactory/saves/RL/Qwen3-8B-Materials-Merged-30", help="模型路径")
    parser.add_argument("--data_path", type=str, default="/home/user_liang/materials/LlamaFactory/test/test_v2.jsonl", help="测试数据路径")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="最大生成长度")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="显存占用比例(0.0-1.0)")
    parser.add_argument("--output_path", type=str, default="/home/user_liang/materials/LlamaFactory/test/evaluate_result_RL/evaluation_predictions_vllm_30_v2.jsonl")
    return parser.parse_args()

def extract_label(text):
    """提取标签逻辑，优先看 </think> 之后的内容"""
    if "</think>" in text:
        content = text.split("</think>")[-1]
    else:
        content = text
    
    # 优先精确匹配
    for label in LABELS:
        if label in content:
            return label
    # 全文模糊匹配兜底
    for label in LABELS:
        if label in text:
            return label
    return "Unknown"

def main():
    args = parse_args()

    # 1. 加载数据
    print(f"正在加载数据: {args.data_path}")
    data = []
    with open(args.data_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except:
                    pass
    
    print(f"共加载 {len(data)} 条测试样本")

    # 2. 准备 Prompts (应用 Chat Template)
    # vLLM 也可以直接吃 token_ids，为了保证和训练一致，我们先用 tokenizer 转换
    print(f"正在准备 Prompts...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    prompts = []
    raw_prompts = []
    y_true = []

    for item in data:
        # 记录真实标签
        y_true.append(extract_label(item['response']))
        raw_prompts.append(item['prompt'])

        # 构建对话
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": item['prompt']}
        ]
        # 转换为 input_ids 或 纯文本 prompt (这里生成纯文本给 vLLM)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # 3. 初始化 vLLM
    print(f"正在初始化 vLLM 引擎...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,  # 如果你是单卡跑，设为1；如果是双卡跑且显存不够，可以设为2
        gpu_memory_utilization=args.gpu_memory_utilization, # 防止爆显存
        dtype="bfloat16"
    )

    # 4. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0,           # 贪婪解码，保证结果确定性
        max_tokens=args.max_new_tokens,
        stop=["<|endoftext|>", "<|im_end|>"] # 碰到这些特殊token停止
    )

    # 5. 开始极速推理
    print(f"开始 vLLM 推理 (Total: {len(prompts)})...")
    # vLLM 的 generate 是全自动并行的，不需要 tqdm 循环 batch
    outputs = llm.generate(prompts, sampling_params)

    # 6. 解析结果
    y_pred = []
    results_to_save = []

    # outputs 的顺序和 prompts 的输入顺序是一一对应的
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        pred_label = extract_label(generated_text)
        
        y_pred.append(pred_label)
        
        results_to_save.append({
            "prompt": raw_prompts[i],
            "raw_output": generated_text,
            "pred": pred_label,
            "true": y_true[i]
        })

    # 7. 计算指标
    print("\n" + "="*50)
    print("EVALUATION RESULTS (vLLM Accelerated)")
    print("="*50)

    # 过滤无效数据
    valid_indices = [i for i, p in enumerate(y_pred) if p != "Unknown"]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if len(valid_indices) > 0:
        print(f"Accuracy:    {accuracy_score(y_true_valid, y_pred_valid):.4f}")
        print(f"Macro F1:    {f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_valid, y_pred_valid, zero_division=0))
    else:
        print("Warning: 没有解析出有效标签！")

    # 8. 保存
    out_path = Path(args.output_path)
    with out_path.open("w", encoding="utf-8") as f:
        for res in results_to_save:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"结果已保存至: {out_path}")

if __name__ == "__main__":
    main()