import math
import zlib
import yaml


class CognitiveEngine:
    """
    AiTokenROI拟合计算，无多余的侵入性操作，不修改、裁剪任何Prompt或者对话内容，无RAG检索
    """

    def __init__(self, max_context: int = 128000, architecture: str = "hybrid", gamma: float = 0.25):
        self.max_context = max_context
        self.architecture = architecture
        self.gamma = gamma if architecture == "hybrid" else 0.0
        self.CHUNK_SIZE = 128  # 恒定步长，保证积分收敛

        with open("./api.yml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.T_HEAD_SAFE = config["T_HEAD_SAFE"]
        self.T_TAIL_SAFE = config["T_TAIL_SAFE"]

    def _get_dynamic_params(self, t_obs: int) -> dict:
        if t_obs <= min(self.T_HEAD_SAFE, self.T_TAIL_SAFE):
            return {"alpha": 1.0, "beta": 1.0, "k1": 0.0, "k2": 0.0}

        x_half_head = max(self.T_HEAD_SAFE / t_obs, 0.01)
        x_half_tail = max(self.T_TAIL_SAFE / t_obs, 0.01)

        ln_2 = 0.693147
        return {
            "alpha": 1.0,
            "beta": 1.0 + math.log1p(min(t_obs / self.max_context, 1.0)),
            "k1": ln_2 / (x_half_head ** 2),
            "k2": ln_2 / (x_half_tail ** 2)
        }

    @staticmethod
    def _integrate_gaussian(a: float, b: float, k: float, coef: float, is_tail: bool = False) -> float:
        """拟合注意力遗忘区和有效 Token 计算"""
        if k == 0 or a >= b:
            return coef * (b - a)

        integral_coef = math.sqrt(math.pi) / (2.0 * math.sqrt(k))

        if not is_tail:
            return coef * integral_coef * (math.erf(b * math.sqrt(k)) - math.erf(a * math.sqrt(k)))
        else:
            return coef * integral_coef * (math.erf((1.0 - a) * math.sqrt(k)) - math.erf((1.0 - b) * math.sqrt(k)))

    @staticmethod
    def _find_attention_valley(k1: float, k2: float, beta: float) -> float:
        """注意力遗忘区内的最低点坐标计算"""
        if abs(k1 - k2) < 1e-9:
            return max(0.0, min(1.0, 0.5 - math.log(beta) / (2 * max(k2, 1e-9))))

        a, b_param, c = k1 - k2, 2 * k2, math.log(beta) - k2
        delta = b_param ** 2 - 4 * a * c

        if delta >= 0:
            roots = [r for r in ((-b_param + math.sqrt(delta)) / (2 * a), (-b_param - math.sqrt(delta)) / (2 * a)) if
                     0.0 <= r <= 1.0]
            if roots:
                return roots[0]
        return 0.5

    @staticmethod
    def _build_token_manifold(full_text: str) -> tuple[list[float], int]:
        """Token 压缩映射，模拟分词器算法"""
        total_chars = len(full_text)
        if total_chars == 0:
            return [], 0

        cdf = [0.0] * total_chars
        current_token_acc = 0.0

        consecutive_spaces = 0
        consecutive_alnum = 0

        for i, char in enumerate(full_text):
            byte_len = len(char.encode('utf-8'))
            weight = 0.0

            if char.isspace():
                consecutive_alnum = 0
                consecutive_spaces += 1
                # 拟合 BBPE 的连续空格合并
                weight = 1.0 / math.log(math.e + consecutive_spaces * 2.0)
            elif char.isalnum() and byte_len == 1:
                consecutive_spaces = 0
                consecutive_alnum += 1
                # 拟合英文压缩率
                weight = max(0.25, 1.0 / math.log(math.e + consecutive_alnum))
            else:
                consecutive_spaces = 0
                consecutive_alnum = 0
                if byte_len == 3:
                    weight = 0.8  # 拟合中文压缩率，CJK 的基础权重
                else:
                    weight = byte_len * 0.5

            current_token_acc += weight
            cdf[i] = current_token_acc

        estimated_total_tokens = int(current_token_acc)

        # 归一化至[0,1]
        for i in range(total_chars):
            cdf[i] = cdf[i] / current_token_acc if current_token_acc > 0 else 0.0

        return cdf, estimated_total_tokens

    @staticmethod
    def _find_dead_zone(slices: list, theta: float = 0.2) -> dict:
        """Kadane 算法寻找最大连续无效 Token 存在区间"""
        max_so_far = 0.0
        current_max = 0.0
        best_start_idx = 0
        best_end_idx = -1
        current_start_idx = 0

        for i, s in enumerate(slices):
            v_k = s['eff_t'] / s['base_t'] if s['base_t'] > 0 else 1.0
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
            dead_base = sum(s['base_t'] for s in slices[best_start_idx:best_end_idx + 1])
            dead_eff = sum(s['eff_t'] for s in slices[best_start_idx:best_end_idx + 1])
            return {
                "start_x": slices[best_start_idx]['start_x'],
                "end_x": slices[best_end_idx]['end_x'],
                "dead_tokens": int(dead_base),
                "wasted_tokens": int(dead_base - dead_eff)
            }

        return {"start_x": 0.0, "end_x": 0.0, "dead_tokens": 0, "wasted_tokens": 0}

    def calculate(self, full_text: str, t_obs: int = 0) -> dict:
        if not full_text:
            return {"t_eff": 0, "bubble_rate": 0.0, "arch_bubble_rate": 0.0, "rn": 0.0, "params": {}}

        cdf, est_total_tokens = self._build_token_manifold(full_text)
        actual_t_obs = t_obs if t_obs > 0 else est_total_tokens

        if actual_t_obs <= 0:
            return {"t_eff": 0, "bubble_rate": 0.0, "arch_bubble_rate": 0.0, "rn": 0.0, "params": {}}

        p = self._get_dynamic_params(actual_t_obs)
        x_min = self._find_attention_valley(p["k1"], p["k2"], p["beta"])

        # 状态 Zlib 计算信息压缩度
        compressor = zlib.compressobj(level=1)

        total_len = len(full_text)
        t_eff_arch_total = 0.0
        t_eff_semantic_total = 0.0
        slice_records = []

        # 定长步进扫描，维持微积分 δ_x 恒定
        for i in range(0, total_len, self.CHUNK_SIZE):
            chunk = full_text[i: i + self.CHUNK_SIZE]

            a = cdf[i] if i > 0 else 0.0
            end_idx = min(i + self.CHUNK_SIZE, total_len - 1)
            b = cdf[end_idx]

            if a >= b:
                continue

            interval_len = b - a

            chunk_bytes = chunk.encode('utf-8')
            orig_bytes_len = max(len(chunk_bytes), 1)
            compressed_chunk_bytes = compressor.compress(chunk_bytes) + compressor.flush(zlib.Z_SYNC_FLUSH)

            # 扣除 Z_SYNC_FLUSH 基础开销
            net_delta_len = max(0, len(compressed_chunk_bytes) - 5)
            density = min(1.0, net_delta_len / orig_bytes_len)

            ssm_area = 0.0
            if b <= x_min:
                ssm_area = self._integrate_gaussian(a, b, p["k1"], p["alpha"], is_tail=False)
            elif a >= x_min:
                ssm_area = self._integrate_gaussian(a, b, p["k2"], p["beta"], is_tail=True)
            else:
                ssm_area = (self._integrate_gaussian(a, x_min, p["k1"], p["alpha"], is_tail=False) +
                            self._integrate_gaussian(x_min, b, p["k2"], p["beta"], is_tail=True))

            ssm_area = min(interval_len, ssm_area)

            # Transformer 均匀覆盖概率叠加
            hybrid_arch_area = (1.0 - self.gamma) * ssm_area + self.gamma * interval_len
            arch_valid_tokens = actual_t_obs * hybrid_arch_area

            # 计算真实语义 Token
            slice_eff_semantic = arch_valid_tokens * density

            t_eff_arch_total += arch_valid_tokens
            t_eff_semantic_total += slice_eff_semantic

            slice_records.append({
                "start_x": a,
                "end_x": b,
                "base_t": actual_t_obs * interval_len,
                "eff_t": slice_eff_semantic
            })

        t_eff_semantic_total = int(t_eff_semantic_total)
        t_eff_arch_total = int(t_eff_arch_total)
        final_rn = max(0, actual_t_obs - t_eff_semantic_total)

        dead_zone = self._find_dead_zone(slice_records, theta=0.2)

        return {
            "总观测_Token": actual_t_obs,
            "最终语义有效_Token": t_eff_semantic_total,
            "架构理论留存_Token": t_eff_arch_total,
            "总认知泡沫量": float(final_rn),
            "全局泡沫率": final_rn / max(1, actual_t_obs),
            "纯架构缺陷泡沫率": max(0.0, (actual_t_obs - t_eff_arch_total) / max(1, actual_t_obs)),
            "动力学推导参数": p,
            "混合架构_Gamma_权重": self.gamma,
            "深度拓扑诊断": {
                "注意力谷底坐标": x_min,
                "认知死区起点": dead_zone["start_x"],
                "认知死区终点": dead_zone["end_x"],
                "死区沉没成本_Token": dead_zone.get("wasted_tokens", 0)
            }
        }
