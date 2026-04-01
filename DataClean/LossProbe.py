import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # 1. 基础配置与目录（复用你的 embedding 路径）
    model_path = "/home/wangchanghui/ai/models/Qwen2.5-1.5B-Base"
    jsonl_dir = "/home/wangchanghui/pycharm_projects/Ai_OCR/data/"
    output_dir = "/home/wangchanghui/ai/loss"
    temp_dir = "/home/wangchanghui/ai/temp"

    state_file = os.path.join(temp_dir, "probe_state.json")
    output_parquet = os.path.join(output_dir,"result-loss-proxy.parquet")

    MAX_TOKENS = 12800
    NUM_BUCKETS = 100
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 2. 初始化或恢复状态
    processed_files = set()
    global_buckets = {str(i): {"sum_loss": 0.0, "count": 0} for i in range(NUM_BUCKETS)}
    processed_docs_total = 0

    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            processed_files = set(checkpoint.get("processed_files", []))
            global_buckets = checkpoint.get("global_buckets", global_buckets)
            processed_docs_total = checkpoint.get("processed_docs_total", 0)
        print(f"检测到存档，已跳过 {len(processed_files)} 个文件，已累积 {processed_docs_total} 篇文档。")

    # 3. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        load_in_4bit=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    jsonl_files = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))

    with torch.no_grad():
        for file_path in jsonl_files:
            file_name = os.path.basename(file_path)
            if file_name in processed_files:
                continue

            print(f"正在处理新文件: {file_name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        full_text = data.get("context", "") + "\n\n" + data.get("input", "")
                        if len(full_text.strip()) < 1000: continue

                        tokens = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)[
                            "input_ids"].to(model.device)
                        if tokens.shape[1] < 2000: continue

                        outputs = model(tokens)
                        shift_logits = outputs.logits[..., :-1, :].contiguous()
                        shift_labels = tokens[..., 1:].contiguous()

                        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                                shift_labels.view(-1)).cpu().numpy()

                        # 归并入统计桶
                        bucket_size = len(token_losses) / NUM_BUCKETS
                        for i in range(NUM_BUCKETS):
                            start_idx = int(i * bucket_size)
                            end_idx = int((i + 1) * bucket_size) if i < NUM_BUCKETS - 1 else len(token_losses)
                            if start_idx < end_idx:
                                global_buckets[str(i)]["sum_loss"] += float(np.mean(token_losses[start_idx:end_idx]))
                                global_buckets[str(i)]["count"] += 1

                        processed_docs_total += 1
                        del tokens, outputs, shift_logits, shift_labels, token_losses
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"处理文档时跳过错误: {e}")
                        continue

            # 4. 原子性保存进度（每处理完一个文件保存一次）
            processed_files.add(file_name)
            temp_state_path = state_file + ".tmp"
            with open(temp_state_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "processed_files": list(processed_files),
                    "global_buckets": global_buckets,
                    "processed_docs_total": processed_docs_total
                }, f, ensure_ascii=False, indent=2)
            os.replace(temp_state_path, state_file)  # 原子重命名
            print(f"文件 {file_name} 处理完毕，进度已存档至 temp 目录。")

    # 5. 生成最终结果
    print("\n所有文件处理完成，生成最终平滑 Loss 曲线...")
    records = []
    avg_obs_tokens = int(MAX_TOKENS * 0.8)

    for i in range(NUM_BUCKETS):
        bucket_data = global_buckets[str(i)]
        if bucket_data["count"] > 0:
            records.append({
                "pos_ratio": float((i + 0.5) / NUM_BUCKETS),
                "local_loss": float(bucket_data["sum_loss"] / bucket_data["count"]),
                "total_obs_tokens": avg_obs_tokens
            })

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_parquet, compression='zstd')
    print(f"统计完成。Parquet 已存入: {output_parquet}")


if __name__ == "__main__":
    main()