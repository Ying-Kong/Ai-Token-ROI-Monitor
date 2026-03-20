import math
import zlib
import re
from collections import deque


class CognitiveEngine:
    """
    AiTokenROI拟合计算，不加载tiktoken和Embedding模型，无语义匹配，无多余的侵入性操作，不修改、裁剪、增添任何Prompt或者对话内容

    核心逻辑: 通过“注意力差”与“信息熵密度”拟合真实Token利用率

    1. 泡沫Token: 由于模型架构和文本冗余导致的无效Token
    2. 计算公式: Token = ∑ (片段长度 * 注意力权重 * Zlib 压缩密度),
    $$Token_{eff} = \sum (L_i \cdot W_{attn, i} \cdot D_{density, i})$$
    3. 拟合计算引擎基于 Transformer 物理特性建模。修改参数仅会扭曲度量结果，无法修改模型实际的认知坍塌规律

    """
    # 唯一对外定义的元数据，用于存证
    ALGO_IDENTITY = "CognitiveROI: Entropy-based Attention Audit"


    def __init__(self, max_context: int = 128000, architecture: str = "hybrid", gamma: float = 0.25):
        self.max_context = max_context
        self.architecture = architecture
        self.gamma = gamma if architecture == "hybrid" else 0.0
        self.CHUNK_SIZE = 128
        self.T_HEAD_SAFE = 16000
        self.T_TAIL_SAFE = 12000

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

        a, b_param, c = k2 - k1, 2 * k2, math.log(beta) - k2
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

        for match in re.finditer(r'(\s+)|([a-zA-Z0-9]+)|([^\sa-zA-Z0-9]+)', full_text):
            start, end = match.span()
            spaces, alnums, others = match.groups()

            length = end - start

            if spaces:
                for i in range(length):
                    current_token_acc += 0.05 + 0.95 * math.exp(-0.5 * (i + 1))
                    cdf[start + i] = current_token_acc

            elif alnums:
                FIXED_ALNUM_WEIGHT = 0.30

                for i in range(length):
                    weight_val = max(FIXED_ALNUM_WEIGHT, 0.25 + 0.75 * math.exp(-0.5 * (i + 1)))

                    current_token_acc += weight_val
                    cdf[start + i] = current_token_acc
            elif others:
                encoded_bytes_len = len(others.encode('utf-8'))
                avg_bytes = encoded_bytes_len / length

                # 依据平均字节数快速断定 CJK 或 特殊符号
                if avg_bytes >= 2.5:
                    # 判定为 CJK, UTF-8下汉字通常为 3 bytes
                    # 1.32为最新LLM的平均 Token, 1字符 ≈ 1.32Token
                    weight = 1.32
                elif avg_bytes >= 1.5:
                    # 针对双字节字符（如拉丁扩展、部分符号、俄语等）
                    weight = 1.1
                else:
                    # 针对单字节特殊符号
                    weight = 0.55
                for i in range(length):
                    current_token_acc += weight
                    cdf[start + i] = current_token_acc

        estimated_total_tokens = int(current_token_acc)

        # 归一化至[0,1]
        if current_token_acc > 0:
            inv_acc = 1.0 / current_token_acc
            cdf = [val * inv_acc for val in cdf]

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

    @staticmethod
    def _compute_simhash(text: str) -> int:
        """轻量级 SimHash 计算：将文本映射为 64-bit 语义指纹"""
        if not text:
            return 0

        v = [0] * 64
        encoded_text = memoryview(text.encode('utf-8'))
        length = len(encoded_text)
        # 使用 3-gram 切片提取局部特征
        for i in range(max(1, len(text) - 2)):
            gram = encoded_text[i:i + 3] if len(text) >= 3 else encoded_text
            h = zlib.crc32(gram)
            for j in range(64):
                if (h >> j) & 1:
                    v[j] += 1
                else:
                    v[j] -= 1

        fingerprint = 0
        for j in range(64):
            if v[j] > 0:
                fingerprint |= (1 << j)
        return fingerprint

    @staticmethod
    def _hamming_distance(h1: int, h2: int) -> int:
        return (h1 ^ h2).bit_count()

    def calculate(self, full_text: str, t_obs: int = 0, ssm_lambda: float = None) -> dict:
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
        total_chunks = actual_t_obs / self.CHUNK_SIZE
        horizon_size = max(64, int(total_chunks * 0.2))
        history_horizon = deque(maxlen=horizon_size)
        GLOBAL_PENALTY = 0.45

        # 定长步进扫描，维持微积分 δ_x 恒定
        for i in range(0, total_len, self.CHUNK_SIZE):
            chunk = full_text[i: i + self.CHUNK_SIZE]

            a = cdf[i - 1] if i > 0 else 0.0
            end_idx = min(i + self.CHUNK_SIZE - 1, total_len - 1)
            b = cdf[end_idx]

            if a >= b:
                continue

            interval_len = b - a
            chunk_bytes = chunk.encode('utf-8')
            orig_bytes_len = max(len(chunk_bytes), 1)
            compressed_chunk_bytes = compressor.compress(chunk_bytes) + compressor.flush(zlib.Z_SYNC_FLUSH)

            # 扣除 z_sync_flush 基础开销
            net_delta_len = max(0, len(compressed_chunk_bytes) - 5)
            density = min(1.0, net_delta_len / orig_bytes_len)

            pure_chars = "".join([c for c in chunk if c.isalnum()])
            if len(pure_chars) >= 32:
                chunk_simhash = self._compute_simhash(pure_chars)
                is_redundant = False

                for hist_hash in history_horizon:
                    if self._hamming_distance(chunk_simhash, hist_hash) <= 3:
                        is_redundant = True
                        break

                if is_redundant:
                    density *= GLOBAL_PENALTY  # 补充Zlib 32K的范围限制，Hash全局命中重复信息，大幅降低信息密度
                else:
                    # 限制大小，防止超长上下文带来的 O(N^2)
                    history_horizon.append(chunk_simhash)

            # ssm_lambda逻辑判定
            if ssm_lambda is None:
                # 根据当前观测长度与最大窗口的比例，自动计算物理坍塌系数
                ssm_lambda = 3.0 * (actual_t_obs / self.max_context) ** 2
            else:
                # 根据特定模型手动指定参数
                pass

            area = 0.0
            if b <= x_min:
                area = self._integrate_gaussian(a, b, p["k1"], p["alpha"], is_tail=False)
            elif a >= x_min:
                area = self._integrate_gaussian(a, b, p["k2"], p["beta"], is_tail=True)
            else:
                area = (self._integrate_gaussian(a, x_min, p["k1"], p["alpha"], is_tail=False) +
                            self._integrate_gaussian(x_min, b, p["k2"], p["beta"], is_tail=True))

            transformer_prob = min(1.0, area / interval_len) if interval_len > 0 else 0.0
            ssm_prob = math.exp(-ssm_lambda * (1.0 - b))

            # Transformer 均匀覆盖概率叠加
            hybrid_retention_prob = (1.0 - self.gamma) * transformer_prob + self.gamma * ssm_prob
            arch_valid_tokens = actual_t_obs * interval_len * hybrid_retention_prob

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
        t_eff_arch_total = round(t_eff_arch_total)
        final_rn = max(0, actual_t_obs - t_eff_semantic_total)

        dead_zone = self._find_dead_zone(slice_records, theta=0.2)

        return {
            "总观测_Token": actual_t_obs, # API JSON返回的全部 Token
            "最终语义有效_Token": t_eff_semantic_total, # 模型真正理解并思考输出的有意义 Token
            "架构理论留存_Token": t_eff_arch_total, # LLM始终记忆的 Token
            "总认知泡沫量": float(final_rn), # 低信息熵的 Token 总数
            "全局泡沫率": final_rn / max(1, actual_t_obs), # LLM由于冗余对话，导致的泡沫 Token 在 All_Token 内的占比
            "纯架构缺陷泡沫率": max(0.0, (actual_t_obs - t_eff_arch_total) / max(1, actual_t_obs)), # LLM的底层架构造成的 Token 损失
            "动力学推导参数": p,
            # 包含α, β, k1, k2
            # α: 头部权重系数
            # β: 尾部增强系数
            # k1: 头部注意力衰减率
            # k2: 尾部注意力衰减度
            "混合架构_Gamma_权重": self.gamma, # 混合架构（Transformer和SSM）的拟合参数
            "深度拓扑诊断": {
                "注意力谷底坐标": x_min,
                # [start_x,end_x]，LLM在全局对话中注意力遗失的区域
                "认知死区起点": dead_zone["start_x"],
                "认知死区终点": dead_zone["end_x"],
                # 在遗忘区内被计算的无用 Token
                "死区沉没成本_Token": dead_zone.get("wasted_tokens", 0)
            }
        }
