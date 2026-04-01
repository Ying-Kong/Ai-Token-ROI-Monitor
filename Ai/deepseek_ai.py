import datetime
import yaml
import os
from openai import OpenAI
from Ai.engine import CognitiveEngine


def get_persona() -> dict:
    with open("persona.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config() -> dict:
    with open("api.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deepseek(user_question: str) -> None:
    config = get_config()
    persona = get_persona()

    api_key = config.get("api_key")
    base_url = config.get("base_url")
    model_name = config.get("model", "deepseek-chat")
    max_ctx = config.get("max_context", 128000)

    user_name = persona.get("user_name", "好朋友")

    system_content = persona.get("system_prompt", "").format(user=user_name)
    instruction = persona.get("task_instruction", "").format(user=user_name)

    # 初始化认知引擎
    engine = CognitiveEngine(max_context=max_ctx, architecture="hybrid", gamma=0.25)

    client = OpenAI(api_key=api_key, base_url=base_url)

    target_file = "test.txt"
    log_file = "test.log"

    # 读取长文本背景
    if not os.path.exists(target_file):
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("")

    with open(target_file, "r", encoding="utf-8") as f:
        user_content_history = f.read().strip()

    latest_prompt_content = f"{instruction}\n\n{user_question}"

    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user',
         'content': f"{user_content_history}\n{latest_prompt_content}" if user_content_history else latest_prompt_content},
    ]

    # 模拟底层的 Chat Template 物理边界，以确保 Zlib 探针隔离性
    full_text_for_engine = f"[SYSTEM]\n{system_content}\n[HISTORY]\n{user_content_history}\n[LATEST]\n{latest_prompt_content}"

    # 发起流式 API 请求
    response = client.chat.completions.create(
        model=model_name,
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

        if chunk.usage is not None:
            t_obs = chunk.usage.prompt_tokens

    print("\n" + "-" * 30)

    # 持久化响应内容
    if full_response_text:
        with open(target_file, "a", encoding="utf-8") as f:
            # 增加换行符避免与下一次历史读取粘连
            f.write(f"\n[用户]: {user_question}\n")
            f.write(f"[AI]: {full_response_text}\n")
        print(f"响应内容已自动追加至文档: {target_file}")

    # 调用 CognitiveEngine 进行 ROI 审计 (传入大模型真实返回的 Prompt Token 耗量)
    metrics = engine.calculate(full_text_for_engine, t_obs)

    # 安全地写入日志，彻底修复作用域异常与中文键名报错
    if t_obs > 0:
        round_num = 1
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as lf:
                lines = [line for line in lf.readlines() if line.strip()]
                round_num = len(lines) + 1

        diag = metrics.get("深度拓扑诊断", {})
        params = metrics.get("动力学推导参数", {})

        log_entry = (
            f"轮次:[{round_num:>4}] | "
            f"时间:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Token:[总Token:{t_obs:>6} 有效Token:{metrics.get('最终语义有效_Token', 0):>6}] | "
            f"泡沫率:{metrics.get('全局泡沫率', 0.0):>7.2%} | "
            f"架构溢损:{metrics.get('纯架构缺陷泡沫率', 0.0):>7.2%} | "
            f"溢出点:[x={diag.get('注意力谷底坐标', 0.0):>4.2f}] | "
            f"死区:[{diag.get('认知死区起点', 0.0):>4.2f}至{diag.get('认知死区终点', 0.0):>4.2f}] | "
            f"沉没Token:{diag.get('死区沉没成本_Token', 0):>6} | "
            f"衰减率:[k1={params.get('k1', 0.0):>5.3f}/k2={params.get('k2', 0.0):>5.3f}]\n"
        )

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(log_entry)

        print(f"有效 Token 占比 {(1 - metrics.get('全局泡沫率', 0.0)):.2%}")
        print(f"日志已记录至: {log_file}")