# Ai-Token-ROI-Monitor
Stop paying for what the AI is ignoring！ 不再为被无视的Token买单！
LLM 长文本Token泡沫率审计工具

## 简介
在如今的超长上下文AI时代，大模型即便拥有1M的上下文窗口，其注意力分布也并非完全均匀，由于Lost in the Middle现象及KV Cache压缩策略，大量位于上下文窗口中部的Token虽然被计费，但对模型最终输出的贡献微乎其微
Ai-Token-ROI-Monitor是一款极轻量级(5KB)的token-ROI审计引擎，不附带重型第三方库，不干预用户Prompt，只客观告知使用者当前对话中有多少钱是花在无效Token上的

## 数学推导
项目摒弃了昂贵的语义向量计算，采用拟合的思想，通过三大核心机制对Token有效性进行黑盒审计：1. 动态 U 型注意力权重拟合 (U-Shaped Weighting)根据 Transformer 架构的注意力偏置特性，我们建立两个高斯衰减场。其衰减系数 $k$ 由安全阈值（Safe Threshold）动态推导：$$x_{half} = \max\left(\frac{T_{safe}}{T_{obs}}, 0.05\right)$$$$k = \frac{\ln(2)}{x_{half}^2}$$2. 高斯连续定积分审计 (Gaussian Continuous Integration)为了实现 $O(1)$ 复杂度的实时审计，我们使用误差函数 (Error Function) 对注意力面积进行连续求积，从而获得每个文本切片的理论架构权重 $\omega_{arch}$：$$\omega_{arch} = \int_{a}^{b} \alpha \cdot e^{-k_1 x^2} \, dx + \int_{a}^{b} \beta \cdot e^{-k_2 (1-x)^2} \, dx$$利用 math.erf 实现快速计算：$$\text{Integral}(a, b, k) = \frac{\sqrt{\pi}}{2\sqrt{k}} \left( \text{erf}(b\sqrt{k}) - \text{erf}(a\sqrt{k}) \right)$$3. zlib 信息熵探针 (Semantic Entropy Sensing)通过 zlib 压缩算法测算文本的语义密度 (Density)。高冗余（废话）的文本具有更高的压缩率，其有效系数更低：$$\text{Density} = \min\left(1.0, \frac{\text{Compressed Length}}{\text{Original Length}}\right)$$4. 最终有效 Token 结算 (Final ROI Calculation)$$T_{eff} = \sum_{slice} (T_{obs} \times \omega_{arch\_normalized} \times \text{Density})$$$$\text{Bubble Rate} = 1 - \frac{T_{eff}}{T_{obs}}$$
