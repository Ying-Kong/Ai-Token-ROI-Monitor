import math
import re
import zlib


class CognitiveEngine:
    """
    AI 认知压力与 ROI 审计引擎 (基于纯原生 Python)
    核心机制: U型半衰期动态推导 + zlib 信息熵探针 + math.erf 连续定积分
    """

    def __init__(self, max_context: int = 128000):
        self.max_context = max_context
        # 设定大模型绝对不发生注意力坍塌的安全Token阈值
        self.T_HEAD_SAFE = 16000
        self.T_TAIL_SAFE = 12000

    def _get_dynamic_params(self, t_obs: int) -> dict:
        # 动态参数推导
        # 如果输入在安全区内，不产生任何架构损耗
        if t_obs <= min(self.T_HEAD_SAFE, self.T_TAIL_SAFE):
            return {"alpha": 1.0, "beta": 1.0, "k1": 0.0, "k2": 0.0}

        # 计算安全区占总长度的百分比
        x_half_head = max(self.T_HEAD_SAFE / t_obs, 0.05)
        x_half_tail = max(self.T_TAIL_SAFE / t_obs, 0.05)

        # 半衰期推导: k=ln(2)/x_half^2
        ln_2 = 0.693147
        k1 = ln_2 / (x_half_head ** 2)
        k2 = ln_2 / (x_half_tail ** 2)

        # Beta: 对话越多，模型越倾向关注最后一句话
        utilization = min(t_obs / self.max_context, 1.0)
        beta = 1.0 + math.log1p(utilization)

        return {
            "alpha": 1.0,
            "beta": beta,
            "k1": k1,
            "k2": k2
        }

    """使用误差函数计算定积分面积，O(1)复杂度"""
    @staticmethod
    def _integrate_gaussian(a: float, b: float, k: float, coef: float, is_tail: bool = False) -> float:
        if k == 0:
            return coef * (b - a)

        integral_coef = math.sqrt(math.pi) / (2.0 * math.sqrt(k))

        if not is_tail:
            # 头部曲线积分: int e^{-k x^2} dx
            return coef * integral_coef * (math.erf(b * math.sqrt(k)) - math.erf(a * math.sqrt(k)))
        else:
            # 尾部曲线积分: int e^{-k (1-x)^2} dx
            u_upper = 1.0 - a
            u_lower = 1.0 - b
            return coef * integral_coef * (math.erf(u_upper * math.sqrt(k)) - math.erf(u_lower * math.sqrt(k)))

    def calculate(self, full_text: str, t_obs: int) -> dict:
        """执行全链路认知损耗计算"""
        if not full_text or t_obs <= 0:
            return {"t_eff": 0, "bubble_rate": 0.0, "arch_bubble_rate": 0.0, "rn": 0.0, "params": {}}

        # 1. 获取动态参数
        p = self._get_dynamic_params(t_obs)
        max_possible_weight = max(p["alpha"], p["beta"])

        # 2. 正则不定长语义切片
        # 匹配任何非标点组成的句子，并带上其后的标点
        matches = list(re.finditer(r'[^\n]+\n*', full_text))
        total_len = len(full_text)

        t_eff_arch_total = 0.0  # 仅由于大模型架构留存的 Token
        t_eff_semantic_total = 0.0  # 结合人类废话后，最终真实的有效 Token

        for match in matches:
            chunk = match.group()
            # 获取该句子在物理全文中的位置比例[a, b]
            a = match.start() / total_len
            b = match.end() / total_len

            if a == b: continue

            # zlib探针
            chunk_bytes = chunk.encode('utf-8-sig')
            orig_len = len(chunk_bytes)
            # 使用zlib进行压缩，压缩率越高，无用信息越多
            comp_len = len(zlib.compress(chunk_bytes, level=6))
            # 限制最高密度为1.0
            density = min(1.0, comp_len / max(1, orig_len))

            # 连续定积分计算
            mid_point = (a + b) / 2.0
            h_val = p["alpha"] * math.exp(-p["k1"] * (mid_point ** 2))
            t_val = p["beta"] * math.exp(-p["k2"] * ((1.0 - mid_point) ** 2))

            if h_val > t_val:
                area = self._integrate_gaussian(a, b, p["k1"], p["alpha"], is_tail=False)
            else:
                area = self._integrate_gaussian(a, b, p["k2"], p["beta"], is_tail=True)

            # 归一化面积，确保权重峰值严格 <= 1.0
            normalized_area = area / max_possible_weight

            # 当前切片的物理基础Token
            t_slice_base = t_obs * (b - a)

            # 架构层面保留下来的 Token
            arch_valid_tokens = t_obs * normalized_area
            t_eff_arch_total += arch_valid_tokens

            # 最终的真实有效 Token
            t_eff_semantic_total += arch_valid_tokens * density

        # 3. 数据结算
        t_eff_semantic_total = int(t_eff_semantic_total)
        t_eff_arch_total = int(t_eff_arch_total)

        final_rn = t_obs - t_eff_semantic_total

        return {
            "t_obs": t_obs,
            "t_eff": t_eff_semantic_total,  # 最终实际有效 Token
            "t_eff_arch": t_eff_arch_total,  # 理论架构最高有效 Token
            "rn": float(final_rn),  # 总泡沫量
            "bubble_rate": final_rn / t_obs,  # 总泡沫率
            "arch_bubble_rate": (t_obs - t_eff_arch_total) / t_obs,  # 纯模型架构缺陷导致的泡沫率
            "params": p
        }