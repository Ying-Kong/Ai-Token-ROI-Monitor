import os
import torch
import pandas as pd
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ==========================================
# 1. 数据集定义 (带全局随机抽样)
# ==========================================
class ParquetDataset(Dataset):
    def __init__(self, parquet_dir, sample_size=3200):
        files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        df_list = [pd.read_parquet(f, columns=['chunk_str']) for f in files]
        full_df = pd.concat(df_list).reset_index(drop=True)

        total_len = len(full_df)
        actual_sample_size = min(sample_size, total_len)
        print(f"数据总量: {total_len}，设定的采样量: {actual_sample_size}")

        self.data = full_df['chunk_str'].dropna().sample(n=actual_sample_size, random_state=42).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 2. 核心物理特征提取逻辑
# ==========================================
running_stats = {"sum_gamma": 0.0, "count": 0}
captured_batch_attn = []


def batch_get_attention_hook(module, input, output):
    if len(output) > 1 and output[1] is not None:
        # 立即转移至 CPU 并转为 float16 节省内存
        captured_batch_attn.append(output[1].detach().cpu().to(torch.float16))


def fast_derive_gamma(attn_tensor):
    """从注意力矩阵中直接抓取 gamma (混合架构权重)"""
    # 转换为 float32 进行高精度数学运算
    avg_matrix = torch.mean(attn_tensor, dim=1).to(torch.float32)
    bs, seq_len, _ = avg_matrix.shape

    # 构建底层物理坐标系 (行索引减去列索引，得到相对距离)
    idx = torch.arange(seq_len, device=avg_matrix.device)
    distance_matrix = idx.unsqueeze(1) - idx.unsqueeze(0)

    # 定义 SSM 的局部状态有效窗口
    ssm_window = 16

    # 提取局部衰减带掩码 (严格的下三角局部区域)
    local_mask = (distance_matrix >= 0) & (distance_matrix <= ssm_window)
    local_mask = local_mask.unsqueeze(0).expand(bs, -1, -1).float()

    gamma_list = []
    for i in range(bs):
        # 计算 SSM 模式 (局部状态带) 的物理算力消耗积
        ssm_mass = torch.sum(avg_matrix[i] * local_mask[i])

        # 计算总体算力消耗积
        total_mass = torch.sum(avg_matrix[i])

        # 直接得出 gamma 的底层物理真实比例
        gamma_value = ssm_mass / torch.clamp(total_mass, min=1e-9)
        gamma_list.append(gamma_value.item())

    return gamma_list


# ==========================================
# 3. 主控制流
# ==========================================
def main():
    model_path = "/home/wangchanghui/ai/models/deepseek-r1-7b"
    parquet_dir = "/home/wangchanghui/ai/embedding"
    output_txt_path = "dynamic_params_gamma.txt"

    BATCH_SIZE = 16
    MAX_LEN = 128
    SAMPLE_SIZE = 3200

    print("正在初始化随机抽样数据集...")
    dataset = ParquetDataset(parquet_dir, sample_size=SAMPLE_SIZE)
    # 彻底杜绝 DataLoader 死锁
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    print(f"启动 4-bit 量化...")

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

    print("开始执行特征矩阵提取与 Gamma 推导...")

    with torch.no_grad():
        for i, batch_text in enumerate(dataloader):
            try:
                captured_batch_attn.clear()

                inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True,
                                   max_length=MAX_LEN).to(model.device)

                _ = model(**inputs)

                # 强制同步，防止 GPU 显存碎片堆积
                torch.cuda.synchronize()

                if captured_batch_attn:
                    g_list = fast_derive_gamma(captured_batch_attn[0])
                    running_stats["sum_gamma"] += sum(g_list)
                    running_stats["count"] += len(g_list)

                # 严格的资源回收
                del inputs
                captured_batch_attn.clear()
                torch.cuda.empty_cache()

                # 每处理 10 个 Batch 输出一次状态并清理系统 RAM
                if i % 10 == 0:
                    gc.collect()
                    cur_gamma = running_stats["sum_gamma"] / max(1, running_stats["count"])
                    print(f"进度: {running_stats['count']}/{len(dataset)} | 当前底层物理 Gamma: {cur_gamma:.6f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Batch {i} 发生显存溢出，正在清理资源并跳过...")
                    captured_batch_attn.clear()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    handle.remove()

    if running_stats["count"] > 0:
        final_gamma = running_stats["sum_gamma"] / running_stats["count"]
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(f"gamma_weight={final_gamma:.6f}\n")
        print(f"\nGamma 提取完成，结果已写入 {output_txt_path}")
        print(f"物理对齐的 Gamma 权重: {final_gamma:.6f}")
    else:
        print("\n提取失败，未成功处理任何数据。")


if __name__ == "__main__":
    main()