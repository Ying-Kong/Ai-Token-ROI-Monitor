# Ai-Token-ROI-Monitor
Stop paying for what the AI is ignoring！ 不再为被无视的Token买单！
LLM 长文本Token泡沫率审计工具

## 简介
在如今的超长上下文AI时代，大模型即便拥有1M的上下文窗口，其注意力分布也并非完全均匀，由于Lost in the Middle现象及KV Cache压缩策略，大量位于上下文窗口中部的Token虽然被计费，但对模型最终输出的贡献微乎其微
Ai-Token-ROI-Monitor是一款极轻量级(5KB)的token-ROI审计引擎，不附带重型第三方库，不干预用户Prompt，只客观告知使用者当前对话中有多少钱是花在无效Token上的

## 核心数学推导
