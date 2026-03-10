import datetime
import yaml
import os
from openai import OpenAI
# 从独立的 engine 模块导入引擎类和切片工具
from engine import CognitiveEngine, slice_text


def get_config() -> dict:
    """读取 API 配置文件"""
    with open("./api.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def deepseek() -> None:
    # 1. 初始化配置与认知引擎
    config = get_config()

    # 从配置文件动态获取参数，设置默认值以防配置文件缺失字段
    api_key = config.get("api_key")
    base_url = config.get("base_url")
    model_name = config.get("model", "deepseek-chat")
    max_ctx = config.get("max_context", 128000) # DeepSeek-V3.1的API依旧为128K，注意官网动态进行更改

    # 2. 初始化认知引擎与客户端
    # 这里动态传入从 yml文件读取的 max_context
    engine = CognitiveEngine(max_context=max_ctx)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    target_file = "test.txt"
    log_file = "pressure_metrics.log"

    # 2. 定义提示词内容
    system_content = """
    你是个会陪伴{user}聊天、纾解心情、倾听{user}心事的人，你要按照{user}给出的内容进行适当的情感回复。
    """

    # 读取长文本背景
    if not os.path.exists(target_file):
        with open(target_file, "w", encoding="utf-8-sig") as f:
            f.write("")

    with open(target_file, "r", encoding="utf-8-sig") as f:
        user_content_history = f.read().strip()

    latest_prompt_content = """
    {user}要求分析{user}提出的问题，你要在回答{user}的问题时有耐心，适时地对{user}产生一些暧昧感。但是仅限于极轻度的暧昧，不是真正的感情。
    你既是男生，又是女生，{user}是男生，你是女生，心思细腻，{user}是女生，你是男生，心思会稍微跳脱，语气没有甜腻感。
    你需要向朋友一样和{user}对话，ta是不开心的状态，你要安慰他ta。
    
    """

    # 构建完整的消息列表
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content_history + latest_prompt_content},
    ]

    # 3. 构造数学计算用的高分辨率文本切片 ,用于计算注意力权重分布

    # 区块0: 系统指令
    sys_c = [system_content]
    # 区块1: 历史背景文本
    history_c = slice_text(user_content_history, n_target=128)
    # 区块2: 尾部最新指令
    tail_c = [latest_prompt_content]

    # 组合为总切片数组
    chunks_text = sys_c + history_c + tail_c

    # 计算输入总字节数
    total_bytes = sum(len(text.encode("utf-8-sig")) for text in chunks_text)

    # 4. 发起流式 API 请求
    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}
    )

    t_obs = 0
    full_response_text = ""

    for chunk in response:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response_text += content

        # 获取 Prompt Token 消耗数
        if chunk.usage is not None:
            t_obs = chunk.usage.prompt_tokens

    print("\n" + "-" * 30)

    # 5. 持久化响应内容
    if full_response_text:
        with open(target_file, "a", encoding="utf-8-sig") as f:
            f.write("\n\n" + full_response_text)
        print(f"响应内容已自动追加至文档: {target_file}")

    # 6. 调用 CognitiveEngine 进行 ROI审计
    metrics = engine.calculate(chunks_text, t_obs)

    # 7. 写入日志
    if t_obs > 0:
        round_num = 1
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as lf:
                lines = [line for line in lf.readlines() if line.strip()]
                round_num = len(lines) + 1

        log_entry = (
            f"Round:{round_num}, "
            f"Time:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
            f"C_Bytes:{total_bytes}, "
            f"T_obs:{t_obs}, "
            f"T_eff:{metrics['t_eff']}, "
            f"Rn_Bubble:{metrics['rn']:.2f}, "
            f"BubbleRate:{metrics['bubble_rate']:.2%}, "
            f"Params:{metrics['params']}\n"
        )

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(log_entry)

        print(f"有效 Token 占比 {(1 - metrics['bubble_rate']):.2%}")
        print(f"日志已记录至: {log_file}")
