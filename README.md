# 🛡️ Ai-Token-ROI-Monitor
Stop paying for what the AI is ignoring！ 不再为被无视的Token买单！
LLM 长文本Token泡沫率审计工具

## 📖 简介
在如今的超长上下文AI时代，大模型即便拥有1M的上下文窗口，其注意力分布也并非完全均匀，由于Lost in the Middle现象及KV Cache压缩策略，大量位于上下文窗口中部的Token虽然被计费，但对模型最终输出的贡献微乎其微

Ai-Token-ROI-Monitor是一款极轻量级(5KB)的token-ROI审计引擎，不干预用户Prompt，只客观告知使用者当前对话中有多少钱是花在无效Token上

## ✨ 特性
- 零依赖: 仅使用Python原生库，核心代码仅5KB
- 非侵入式: 不修改Prompt，不拦截API请求，仅通过 API返回的usage数据进行事后审计
- 架构感知: 预设DeepSeek-V3.1，Gemini，ChatGPT等主流模型的安全参数，支持YAML动态调整
- 可视化日志: 自动生成日志报表，精确记录架构底噪与内容泡沫

## 🚀 快速开始
1. 安装
直接复制engine.py放入你的项目即可

2.基础用法
```python
from engine import CognitiveEngine

# 初始化，设置最大上下文窗口
engine = CognitiveEngine(max_context=128000)

# 执行审计
# full_text: 发送给 AI 的完整上下文
# t_obs: API 返回的真实 prompt_tokens
metrics = engine.calculate(full_text, t_obs)

print(f"总泡沫率: {metrics['bubble_rate']:.2%}")
print(f"有效 Token: {metrics['t_eff']}"
```
核心逻辑请参阅 engine.py

## 📈 为什么选择token-ROI？
在大模型处理各种任务时，虽然模型在超过有效注意力后仍能答对，但如果LLM对本次的Token有效利用率过低，意味着你支付了大量的api溢价。
本工具并不帮你做决定，它只是在你的控制台挂一个**成本监控球**，让你在挥霍你的Token时，对烧掉的api额度有一个清晰的量化认知。

## 🤝 贡献与交流
欢迎提交 Issue 或 Pull Request 来优化针对不同模型的**T_HEAD_SAFE和T_TAIL_SAFE**值参数。

License: MIT
Author: [Ying-Kong](https://github.com/Ying-Kong))
