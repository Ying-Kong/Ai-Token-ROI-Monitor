import datetime
import yaml
import os
from openai import OpenAI
# 从独立的 engine 模块导入计算引擎
from engine import CognitiveEngine


def get_config() -> dict:
    """读取 API 配置文件"""
    with open("./api.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def deepseek(user_question: str) -> None:
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
    
    """

    # 读取长文本背景
    if not os.path.exists(target_file):
        with open(target_file, "w", encoding="utf-8-sig") as f:
            f.write("")

    with open(target_file, "r", encoding="utf-8-sig") as f:
        user_content_history = f.read().strip()

    # 1. {user_question}为此轮次{user}要说的话

    # 具体针对ai本轮任务的Prompt要求,{user_question}通过main.py的def deepseek()传参
    latest_prompt_content = f"""
    
    {user_question}   
    """

    # 构建完整的消息列表
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content_history + "\n" + latest_prompt_content},
    ]

    # 3. 构造数学计算用的高分辨率文本切片 ,用于计算注意力权重分布
    # 区块0: 系统指令
    full_text_for_engine = f"{system_content}\n{user_content_history}\n{latest_prompt_content}"
    # 计算输入总字节数
    total_bytes = len(full_text_for_engine.encode("utf-8-sig"))

    # 4. 发起流式API请求
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
            f.write(f"\n\n[用户]: {user_question}")
            f.write(f"\n[AI]: {full_response_text}")
        print(f"响应内容已自动追加至文档: {target_file}")

    # 6. 调用 CognitiveEngine 进行 ROI审计
    metrics = engine.calculate(full_text_for_engine, t_obs)

    # 7. 写入日志
    if t_obs > 0:
        round_num = 1
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as lf:
                lines = [line for line in lf.readlines() if line.strip()]
                round_num = len(lines) + 1

                diag = metrics["diagnostics"]

                log_entry = (
                    f"轮次:[{round_num:>4}] | "
                    f"时间:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Token:[总Token:{t_obs:>6} 有效Token:{metrics['t_eff']:>6}] | "
                    f"泡沫率:{metrics['bubble_rate']:>7.2%} | "
                    f"架构溢损:{metrics['arch_bubble_rate']:>7.2%} | "
                    f"溢出点:[x={diag['valley_center']:>4.2f}] | "
                    f"无效荷载区:[{diag['dead_zone_start']:>4.2f}至{diag['dead_zone_end']:>4.2f}] | "
                    f"浪费Token:{diag['dead_zone_wasted']:>6} | "
                    f"注意力衰减值:[{metrics['params']['k1']:>6.3f}/{metrics['params']['k2']:>6.3f}]\n"
                )

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(log_entry)

        print(f"有效 Token 占比 {(1 - metrics['bubble_rate']):.2%}")
        print(f"日志已记录至: {log_file}")
