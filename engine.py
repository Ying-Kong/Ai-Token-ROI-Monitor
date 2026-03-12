import math
import re
import zlib

import yaml


class CognitiveEngine:
    """
    AI 认知压力与 ROI 审计引擎 (基于纯原生 Python)
    核心机制: U型半衰期动态推导 + zlib 信息熵探针 + math.erf 连续定积分
    """

    def __init__(self, max_context: int = 128000):
        self.max_context = max_context
        # 设定大模型绝对不发生注意力坍塌的安全Token阈值
        with open("./api.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.T_HEAD_SAFE = config["T_HEAD_SAFE"]
        self.T_TAIL_SAFE = config["T_TAIL_SAFE"]

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

        slice_records = [] # 初始化切片记录器

        for match in matches:
            chunk = match.group()
            # 获取该句子在物理全文中的位置比例[a, b]
            a = match.start() / total_len
            b = match.end() / total_len

            if a == b: continue

            # zlib探针
            chunk_bytes = chunk.encode('utf-8')
            orig_len = len(chunk_bytes)
            # 使用zlib进行压缩，压缩率越高，无用信息越多
            comp_len = len(zlib.compress(chunk_bytes, level=1))

            net_comp_len = max(0, comp_len - 11)
            # 限制最高密度为1.0
            density = min(1.0, net_comp_len / max(1, orig_len))

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
            slice_eff_semantic = arch_valid_tokens * density

            t_eff_arch_total += arch_valid_tokens
            # 最终的真实有效 Token
            t_eff_semantic_total += slice_eff_semantic

            slice_records.append({
                "start_x": a,
                "end_x": b,
                "base_t": t_slice_base,
                "eff_t": slice_eff_semantic
            })

        # 3. 数据结算
        t_eff_semantic_total = int(t_eff_semantic_total)
        t_eff_arch_total = int(t_eff_arch_total)
        final_rn = t_obs - t_eff_semantic_total

        x_min = self._find_attention_valley(p["k1"], p["k2"], p["beta"])
        dead_zone = self._find_dead_zone(slice_records, theta=0.2)

        return {
            "t_obs": t_obs,
            "t_eff": t_eff_semantic_total,  # 最终实际有效 Token
            "t_eff_arch": t_eff_arch_total,  # 理论架构最高有效 Token
            "rn": float(final_rn),  # 总泡沫量
            "bubble_rate": final_rn / t_obs,  # 总泡沫率
            "arch_bubble_rate": (t_obs - t_eff_arch_total) / t_obs,  # 纯模型架构缺陷导致的泡沫率
            "params": p,
            "diagnostics": {
                "valley_center": x_min,
                "dead_zone_start": dead_zone["start_x"],
                "dead_zone_end": dead_zone["end_x"],
                "dead_zone_wasted": dead_zone["wasted_tokens"]
            }
        }

    @staticmethod
    def _find_attention_valley(k1: float, k2: float, beta: float) -> float:
        """
        求解 U型注意力包络线的全局极小值点物理坐标 (x_min)
        方程: (k1 - k2)x^2 + 2*k2*x - (k2 + ln(beta)) = 0
        """
        if k1 == 0 and k2 == 0:
            return 0.5  # 在安全区内，没有明显谷底，默认中点

        if abs(k1 - k2) < 1e-9:
            # k1 == k2 时的退化一元一次方程
            x_min = 0.5 - math.log(beta) / (2 * k2) if k2 != 0 else 0.5
        else:
            # 求解一元二次方程
            a = k1 - k2
            b = 2 * k2
            c = math.log(beta) - k2
            delta = b ** 2 - 4 * a * c

            if delta >= 0:
                root1 = (-b + math.sqrt(delta)) / (2 * a)
                root2 = (-b - math.sqrt(delta)) / (2 * a)
                # 取位于物理坐标 [0, 1] 内的有效解
                x_min = root1 if 0.0 <= root1 <= 1.0 else root2
            else:
                x_min = 0.5  # 理论防线，正常参数不会走到这里

        return max(0.0, min(1.0, x_min))

    @staticmethod
    def _find_dead_zone(slices: list, theta: float = 0.2) -> dict:
        """
        使用 Kadane 算法动态规划，寻找绝对沉没成本最大的连续物理区间
        theta: 最低容忍 ROI 阈值 (如 0.2，表示低于 20% 有效率即视为死区)
        """
        max_so_far = 0.0
        current_max = 0.0
        best_start_idx = 0
        best_end_idx = -1
        current_start_idx = 0

        for i, s in enumerate(slices):
            # 单个切片的 ROI
            v_k = s['eff_t'] / s['base_t'] if s['base_t'] > 0 else 1.0

            # 收益函数 = 基础Token * (1 - 实际ROI / 容忍阈值)
            # 物理意义：如果一段文本极度无聊 (v_k < theta)，则视为正向发现；如果很有用，产生负向惩罚
            benefit = s['base_t'] * (1.0 - v_k / theta)

            if current_max + benefit < 0:
                current_max = 0.0
                current_start_idx = i + 1
            else:
                current_max += benefit

            if current_max > max_so_far:
                max_so_far = current_max
                best_start_idx = current_start_idx
                best_end_idx = i

        if max_so_far > 0 and best_end_idx >= best_start_idx:
            # 汇总该死区内的统计数据
            dead_base = sum(s['base_t'] for s in slices[best_start_idx:best_end_idx + 1])
            dead_eff = sum(s['eff_t'] for s in slices[best_start_idx:best_end_idx + 1])
            return {
                "start_x": slices[best_start_idx]['start_x'],
                "end_x": slices[best_end_idx]['end_x'],
                "dead_tokens": int(dead_base),
                "wasted_tokens": int(dead_base - dead_eff)
            }

        return {"start_x": 0.0, "end_x": 0.0, "dead_tokens": 0, "wasted_tokens": 0}