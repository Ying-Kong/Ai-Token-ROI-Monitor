import os
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from transformers import LlamaTokenizerFast

input_dir = os.path.expanduser('/home/wangchanghui/ai/embedding')
output_dir = os.path.expanduser('/home/wangchanghui/ai/embedding_output')
temp_dir = os.path.expanduser('/home/wangchanghui/ai/embedding_temp')

embedding_model = "/home/wangchanghui/ai/models/bge-m3"
tokenizer_model = "/home/wangchanghui/ai/models/deepseek-v3-tokenizer"
batch_size = 32

def set_env():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

def gpu_tasks():
    set_env()
    tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_model,
        local_files_only=True,
        trust_remote_code=False
    )
    model = SentenceTransformer(embedding_model, device="cuda")
    all_tasks = sorted([f for f in os.listdir(input_dir) if f.endswith('.parquet')])
    total_tasks = len(all_tasks)

    for idx, task in enumerate(all_tasks):
        input_path = os.path.join(input_dir, task)
        output_name = task.replace("part-", "result-")
        output_path = os.path.join(output_dir, output_name)
        temp_path = os.path.join(temp_dir, output_name + ".tmp")

        if os.path.exists(output_path):
            continue


        try:
            df = pd.read_parquet(input_path)
            chunk_strings = df['chunk_str'].tolist()
            real_tokens = [len(tokenizer.encode(text, add_special_tokens=False)) for text in chunk_strings]

            embeddings = model.encode(
                chunk_strings,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            df['embedding'] = embeddings.tolist()

            table = pa.Table.from_pandas(df)
            pq.write_table(table, temp_path, compression='zstd')
            os.rename(temp_path, output_path)

        except Exception as e:
            print(f"处理 {task} 时发生崩溃: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    # 清理 PyTorch 显存缓存
    torch.cuda.empty_cache()
    gpu_tasks()
    print("所有分片向量化提取完成。")