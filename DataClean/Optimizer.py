import os
import gc
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from Ai.engine import CognitiveEngine
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==========================================
# 阶段 0: 数据预处理与特征萃取 (解决内存与 I/O 瓶颈)
# ==========================================
def preprocess_attention_data(files):
    print("开始预处理 Attention 数据 (只需执行一次)...")
    # 第一遍：寻找全局 min_loss
    min_loss = float('inf')
    for f in files:
        df = pd.read_parquet(f, columns=["local_loss"])
        current_min = df['local_loss'].min()
        if current_min < min_loss:
            min_loss = current_min
        del df
        gc.collect()

    # 第二遍：提取核心特征，丢弃庞大 DataFrame
    t_obs_list, pos_ratio_list, true_prob_list = [], [], []
    for f in files:
        df = pd.read_parquet(f, columns=["pos_ratio", "local_loss", "total_obs_tokens"])
        t_obs_list.append(df['total_obs_tokens'].values.astype(np.int32))
        pos_ratio_list.append(df['pos_ratio'].values.astype(np.float32))
        true_prob_list.append((min_loss / df['local_loss'].values).astype(np.float32))
        del df
        gc.collect()

    # 合并为紧凑的 numpy 数组
    return (
        np.concatenate(t_obs_list),
        np.concatenate(pos_ratio_list),
        np.concatenate(true_prob_list)
    )


def preprocess_embedding_data(files):
    print("开始预处理 Embedding 数据，解决跨文件截断问题...")

    # 临时字典，用于跨文件拼接同一个 doc_id 的特征
    # 结构: { doc_id: {"chunks": [], "embs": []} }
    global_docs = {}

    for f in files:
        df = pd.read_parquet(f, columns=["doc_id", "chunk_str", "embedding"])
        for doc_id, group in df.groupby("doc_id", sort=False):
            if doc_id not in global_docs:
                global_docs[doc_id] = {"chunks": [], "embs": []}

            global_docs[doc_id]["chunks"].extend(group['chunk_str'].tolist())
            global_docs[doc_id]["embs"].extend(group['embedding'].tolist())

        del df
        gc.collect()

    print("跨文件拼接完成，开始计算真实泡沫率...")
    preprocessed_data = []

    for doc_id, data in global_docs.items():
        full_text = "".join(data["chunks"])
        if not full_text.strip():
            continue

        embs = np.stack(data["embs"])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embs_norm = embs / norms

        redundant_count = 0
        for i in range(1, len(embs_norm)):
            window_start = max(0, i - 64)
            sims = np.dot(embs_norm[window_start:i], embs_norm[i])
            if np.max(sims) > 0.85:
                redundant_count += 1

        true_bubble_rate = (redundant_count / len(embs_norm)) * 0.9
        preprocessed_data.append((full_text, true_bubble_rate))

        # 处理完一个立刻清空其内存
        global_docs[doc_id] = None

    del global_docs
    gc.collect()

    return preprocessed_data


# ==========================================
# 模块 A: 注意力动力学 (纯内存并行计算)
# ==========================================
def compute_attention_chunk(params, true_gamma, t_obs_chunk, pos_ratio_chunk, true_prob_chunk):
    head_ratio, tail_ratio = params
    engine_instance = CognitiveEngine(max_context=128000, architecture="hybrid")

    pred_prob_chunk = np.zeros_like(true_prob_chunk)

    for i in range(len(t_obs_chunk)):
        t_obs = t_obs_chunk[i]
        x = pos_ratio_chunk[i]

        engine_instance.T_HEAD_SAFE = int(t_obs * head_ratio)
        engine_instance.T_TAIL_SAFE = int(t_obs * tail_ratio)
        engine_instance.gamma = true_gamma  # 使用物理测定的固定值

        p = engine_instance._get_dynamic_params(t_obs)
        x_min = engine_instance._find_attention_valley(p["k1"], p["k2"], p["beta"])

        delta = 0.001
        a, b = max(0.0, x - delta), min(1.0, x + delta)
        interval_len = b - a

        if b <= x_min:
            area = engine_instance._integrate_gaussian(a, b, p["k1"], p["alpha"], is_tail=False)
        elif a >= x_min:
            area = engine_instance._integrate_gaussian(a, b, p["k2"], p["beta"], is_tail=True)
        else:
            area = (engine_instance._integrate_gaussian(a, x_min, p["k1"], p["alpha"], is_tail=False) +
                    engine_instance._integrate_gaussian(x_min, b, p["k2"], p["beta"], is_tail=True))

        transformer_prob = min(1.0, area / interval_len) if interval_len > 0 else 0.0
        ssm_lambda = 3.0 * (t_obs / engine_instance.max_context) ** 2
        ssm_prob = np.exp(-ssm_lambda * (1.0 - x))

        # 这里的 true_gamma 已经是常数
        pred_prob_chunk[i] = (1.0 - true_gamma) * transformer_prob + true_gamma * ssm_prob

    chunk_mse = np.sum((pred_prob_chunk - true_prob_chunk) ** 2)
    return chunk_mse, len(t_obs_chunk)


def attention_error_func(params, true_gamma, data_chunks, executor):
    head_ratio, tail_ratio = params
    if head_ratio <= 0 or tail_ratio <= 0: return 1e9

    total_mse = 0.0
    total_rows = 0

    # 将 fixed true_gamma 传入计算集群
    futures = [executor.submit(compute_attention_chunk, params, true_gamma, t, p, tr) for t, p, tr in data_chunks]
    for future in as_completed(futures):
        chunk_mse, rows = future.result()
        total_mse += chunk_mse
        total_rows += rows

    return total_mse / total_rows if total_rows > 0 else 1e9


# ==========================================
# 模块 B: 语义冗余惩罚 (纯内存并行计算)
# ==========================================
def compute_embedding_chunk(params, data_chunk):
    test_penalty = params[0]
    engine_instance = CognitiveEngine(max_context=128000, architecture="hybrid")
    engine_instance.GLOBAL_PENALTY = test_penalty

    chunk_mse = 0.0
    for full_text, true_bubble_rate in data_chunk:
        pred_bubble_rate = engine_instance.calculate(full_text).get("全局泡沫率", 0.0)
        chunk_mse += (pred_bubble_rate - true_bubble_rate) ** 2

    return chunk_mse, len(data_chunk)


def embedding_error_func(params, data_chunks, executor):
    test_penalty = params[0]
    if test_penalty <= 0 or test_penalty > 1.0: return 1e9

    total_mse = 0.0
    total_rows = 0

    futures = [executor.submit(compute_embedding_chunk, params, chunk) for chunk in data_chunks]
    for future in as_completed(futures):
        mse, rows = future.result()
        total_mse += mse
        total_rows += rows

    return total_mse / total_rows if total_rows > 0 else 1e9


# ==========================================
# 主路由执行器
# ==========================================
def main():
    # parquet_dir = "/home/wangchanghui/ai/embedding_output"
    parquet_dir = "/home/wangchanghui/ai/loss"
    files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if
             f.startswith('result-') and f.endswith('.parquet')]

    if not files:
        print("未读取到任何有效的 Parquet 分片文件。")
        return

    # 获取列名判定模式，判完立即清理
    sample_df = pd.read_parquet(files[0])
    columns = sample_df.columns.tolist()
    del sample_df
    gc.collect()

    cpu_cores = os.cpu_count() or 8
    max_workers = max(1, cpu_cores - 2)  # 留2个核给系统，其他全拉满
    print(f"工作进程已配置为 {max_workers} 核并行。")

    if "local_loss" in columns:
        print("检测到 PPL Loss，启动注意力动力学对齐任务...")

        # === 新增：读取已确定的底层物理 Gamma 常量 ===
        true_gamma = 0.25  # 设置一个安全的后备默认值
        gamma_file = "dynamic_params_gamma.txt"
        if os.path.exists(gamma_file):
            with open(gamma_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if "gamma_weight=" in content:
                    true_gamma = float(content.split("=")[1])
            print(f"成功加载物理层 Gamma 常量: {true_gamma:.6f}")
        else:
            print(f"警告：未找到 {gamma_file}，将使用后备默认值 {true_gamma}")
        # ==========================================

        # 1. 执行预处理
        t_obs_all, pos_ratio_all, true_prob_all = preprocess_attention_data(files)

        # 2. 数据切分 (按 CPU 核心数切分)
        chunks_t = np.array_split(t_obs_all, max_workers)
        chunks_p = np.array_split(pos_ratio_all, max_workers)
        chunks_tr = np.array_split(true_prob_all, max_workers)
        data_chunks = list(zip(chunks_t, chunks_p, chunks_tr))

        # === 修改：降维为二维寻优，剔除 Gamma 的初始值和边界 ===
        initial_guess = [0.125, 0.093]
        bounds = [(0.01, 0.5), (0.01, 0.5)]

        print("进入二维优化循环 (已固定 Gamma)...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # === 修改：将 true_gamma 作为静态参数传入 args ===
            result = minimize(
                attention_error_func, initial_guess, args=(true_gamma, data_chunks, executor),
                method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': 50}
            )

        if result.success:
            # === 修改：只接收前两个变量的优化结果 ===
            opt_head, opt_tail = result.x
            with open("dynamic_params_global.txt", "w", encoding="utf-8") as f:
                f.write(f"head_safe_ratio={opt_head:.6f}\n")
                f.write(f"tail_safe_ratio={opt_tail:.6f}\n")
            print("头尾安全区参数修正完毕，已写入 dynamic_params_global.txt")

    elif "embedding" in columns:
        print("检测到高维 Embedding，启动语义冗余惩罚系数对齐任务...")

        # 1. 执行预处理
        preprocessed_data = preprocess_embedding_data(files)

        # 2. 数据切分 (将列表按 CPU 核心数切分)
        chunk_size = max(1, len(preprocessed_data) // max_workers)
        data_chunks = [preprocessed_data[i:i + chunk_size] for i in range(0, len(preprocessed_data), chunk_size)]

        initial_guess = [0.45]
        bounds = [(0.1, 0.9)]

        print("进入优化循环...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            result = minimize(
                embedding_error_func, initial_guess, args=(data_chunks, executor),
                method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': 20}
            )

        if result.success:
            opt_penalty = result.x[0]
            with open("dynamic_params_radio.txt", "a", encoding="utf-8") as f:
                f.write(f"global_penalty={opt_penalty:.6f}\n")
            print("冗余惩罚系数修正完毕，已追加写入 dynamic_params_global.txt")

if __name__ == "__main__":
    main()