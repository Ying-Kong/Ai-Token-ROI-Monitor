import math

def slice_text(text: str, n_target: int = 128) -> list:
    """动态分辨率切割文本"""
    if not text: return []
    total_len = len(text)
    actual_n = max(1, min(n_target, max(1, total_len // 10)))
    step = math.ceil(total_len / actual_n)
    return [text[i: i + step] for i in range(0, total_len, step)]


def _probe_features(chunks: list) -> list:
    """信息熵探针: 防御连续空格等低熵信息攻击"""
    features = []
    for chunk in chunks:
        # 惩罚因子
        eff_len = len(chunk) - (chunk.count(' ') * 0.9)
        features.append(max(eff_len, 0.1))
    return features


class CognitiveEngine:
    """
    AI认知压力强度计算引擎
    """

    def __init__(self, max_context: int = 128000):
        self.max_context = max_context

    def _get_dynamic_params(self, t_obs: int) -> dict:
        """推导当前上下文利用率下的U型衰减参数"""
        utilization = min(t_obs / self.max_context, 1.0)


        return { # 基于个人经验的启发式参数
            "alpha": 1.0,
            "beta": min(0.8 + (0.1 * 3.0), 1.5),  # 尾部权重
            "k1": round(0.5 + 10.0 * (utilization ** 2), 2),
            "k2": round(0.5 + 12.0 * (utilization ** 2), 2)
        }

    def calculate(self, chunks: list, t_obs: int) -> dict:

        if not chunks or t_obs <= 0:
            return {"t_eff": 0, "bubble_rate": 0.0, "rn": 0.0}

        # 1. 参数获取与特征提取
        p = self._get_dynamic_params(t_obs)
        features = _probe_features(chunks)
        total_f = sum(features)
        n = len(chunks)

        # 2. 计算位置权重
        normalized_weights = []
        max_w = max(p["alpha"], p["beta"])
        for i in range(n):
            x = i / (n - 1) if n > 1 else 1.0

            w = p["alpha"] if i == 0 else (p["alpha"] * math.exp(-p["k1"] * x)) + (
                        p["beta"] * math.exp(p["k2"] * (x - 1)))
            normalized_weights.append(w / max_w)

        # 3. 积分求和
        t_valid = 0.0
        for i in range(n):
            t_i = t_obs * (features[i] / total_f)  # 分配局部 Token 密度
            t_valid += t_i * normalized_weights[i]

        t_valid = int(t_valid)
        rn = t_obs - t_valid

        return {
            "t_obs": t_obs,
            "t_eff": t_valid,
            "rn": float(rn),
            "bubble_rate": rn / t_obs if t_obs > 0 else 0.0,
            "params": p  # 返回使用的参数，方便日志审计
        }