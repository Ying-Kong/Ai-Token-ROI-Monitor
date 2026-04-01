import os
import torch
import pandas as pd
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ParquetDataset(Dataset):
    def __init__(self, parquet_dir, sample_size=3200):
        files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        df_list = [pd.read_parquet(f, columns=['chunk_str']) for f in files]
        full_df = pd.concat(df_list).reset_index(drop=True)

        # 核心修改：在全局数据中进行无放回随机抽样，并设定 random_state 保证可复现
        total_len = len(full_df)
        actual_sample_size = min(sample_size, total_len)
        print(f"数据总量: {total_len}，设定的采样量: {actual_sample_size}")

        self.data = full_df['chunk_str'].dropna().sample(n=actual_sample_size, random_state=42).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


running_stats = {"sum_th": 0.0, "count": 0}
captured_batch_attn = []


def batch_get_attention_hook(module, input, output):
    if len(output) > 1 and output[1] is not None:
        captured_batch_attn.append(output[1].detach().cpu().to(torch.float16))


def fast_derive_theta(attn_tensor):
    avg_matrix = torch.mean(attn_tensor, dim=1).to(torch.float32).numpy()
    bs = avg_matrix.shape[0]
    seq_len = avg_matrix.shape[-1]

    th_list = []
    for i in range(bs):
        p = np.clip(avg_matrix[i], 1e-12, 1.0)
        ent = -np.sum(p * np.log2(p), axis=-1) / np.log2(seq_len)
        th_list.append(np.mean(ent))
    return th_list


def main():
    model_path = "/home/wangchanghui/ai/models/deepseek-r1-7b"
    parquet_dir = "/home/wangchanghui/ai/embedding"
    output_txt_path = "dynamic_params_theta.txt"

    # 控制显存峰值与采样规模
    BATCH_SIZE = 16
    MAX_LEN = 128
    SAMPLE_SIZE = 12800  # 抽样数量，3200条足够统计收敛

    print("正在初始化随机抽样数据集...")
    dataset = ParquetDataset(parquet_dir, sample_size=SAMPLE_SIZE)

    # 关闭多进程预读与锁页内存，杜绝死锁
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    print(f"抽样加载完成: {len(dataset)} 条样本。启动 4-bit 量化...")

    conf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", quantization_config=conf,
        trust_remote_code=True, output_attentions=True
    )
    model.eval()

    target_layer = len(model.model.layers) - 1
    handle = model.model.layers[target_layer].self_attn.register_forward_hook(batch_get_attention_hook)

    print("开始执行抽样推理与推导...")

    with torch.no_grad():
        for i, batch_text in enumerate(dataloader):
            try:
                captured_batch_attn.clear()

                inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True,
                                   max_length=MAX_LEN).to(model.device)

                _ = model(**inputs)

                # 阻止 CPU 超前执行，确保 GPU 矩阵计算完全落盘
                torch.cuda.synchronize()

                if captured_batch_attn:
                    th_list = fast_derive_theta(captured_batch_attn[0])
                    running_stats["sum_th"] += sum(th_list)
                    running_stats["count"] += len(th_list)

                # 每一轮执行严格的资源回收
                del inputs
                captured_batch_attn.clear()
                torch.cuda.empty_cache()

                # 增加打印频率，每 10 个 Batch 汇报一次，并回收系统内存
                if i % 10 == 0:
                    gc.collect()
                    cur_th = running_stats["sum_th"] / max(1, running_stats["count"])
                    print(f"进度: {running_stats['count']}/{len(dataset)} | 当前局部死区阈值 Theta: {cur_th:.6f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Batch {i} 发生显存溢出，正在清理资源并跳过该批次...")
                    captured_batch_attn.clear()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    handle.remove()

    if running_stats["count"] > 0:
        final_theta = running_stats["sum_th"] / running_stats["count"]
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(f"theta={final_theta:.6f}\n")
        print(f"\n抽样局部语义分析完成，结果已安全写入 {output_txt_path}")
        print(f"最优 theta: {final_theta:.6f}")
    else:
        print("\n未能成功处理任何数据，未生成 TXT 文件。")


if __name__ == "__main__":
    main()