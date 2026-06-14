"""
贝叶斯优化

用高斯过程建模 prompt 配置与得分的关系，
用期望改进 (EI) 作为采集函数选择下一组参数。
"""

from typing import Dict, List, Optional
import numpy as np
from scipy import stats as sp_stats
from ..core.templates import PromptTemplates
from ..core.builder import PromptBuilder
from ..core.scorer import Scorer


class BayesianSearchResult:
    """贝叶斯优化结果"""

    def __init__(self, best_config: Dict, best_score: float,
                 X_train: np.ndarray, y_train: np.ndarray, history: List[float]):
        self.best_config = best_config
        self.best_score = best_score
        self.X_train = X_train
        self.y_train = y_train
        self.history = history

    def summary(self) -> str:
        templates = PromptTemplates()
        name = templates.config_to_name(self.best_config)
        lines = [
            f"最优配置: {name}",
            f"最优得分: {self.best_score:.2f}",
            f"评估次数: {len(self.y_train)}",
            f"初始最优: {self.history[0]:.2f}",
            f"最终最优: {self.history[-1]:.2f}",
            f"提升: +{self.history[-1] - self.history[0]:.2f}",
        ]
        return "\n".join(lines)


class BayesianSearch:
    """贝叶斯优化搜索"""

    def __init__(self, templates: Optional[PromptTemplates] = None,
                 builder: Optional[PromptBuilder] = None,
                 scorer: Optional[Scorer] = None):
        self.templates = templates or PromptTemplates()
        self.builder = builder or PromptBuilder(self.templates)
        self.scorer = scorer or Scorer()

    def _config_to_vector(self, config: Dict[str, str]) -> np.ndarray:
        """配置编码为数值向量（one-hot）"""
        vec = []
        for param in ["role", "format", "reasoning", "fewshot"]:
            options = self.templates.get_options(param)
            idx = options.index(config[param])
            onehot = [0] * len(options)
            onehot[idx] = 1
            vec.extend(onehot)
        return np.array(vec)

    def _vector_to_config(self, vec: np.ndarray) -> Dict[str, str]:
        """数值向量解码为配置"""
        config = {}
        offset = 0
        for param in ["role", "format", "reasoning", "fewshot"]:
            options = self.templates.get_options(param)
            n = len(options)
            onehot = vec[offset:offset + n]
            idx = int(np.argmax(onehot))
            config[param] = options[idx]
            offset += n
        return config

    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray,
                               best_score: float, xi: float = 0.01) -> np.ndarray:
        """计算期望改进 (EI)"""
        ei = np.zeros_like(mu)
        mask = sigma > 0
        z = (mu[mask] - best_score - xi) / sigma[mask]
        ei[mask] = (mu[mask] - best_score - xi) * sp_stats.norm.cdf(z) + sigma[mask] * sp_stats.norm.pdf(z)
        return np.maximum(ei, 0)

    def _gp_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, length_scale: float = 1.0):
        """简化高斯过程预测（RBF 核）— 向量化版本，使用 solve 替代 inv 提升数值稳定性"""
        # 训练集核矩阵 (向量化)
        dist_train = np.sum((X_train[:, None] - X_train[None, :]) ** 2, axis=-1)
        K = np.exp(-dist_train / (2 * length_scale ** 2)) + np.eye(len(X_train)) * 0.01

        # 测试集与训练集的核 (向量化)
        dist_test = np.sum((X_test[:, None] - X_train[None, :]) ** 2, axis=-1)
        K_star = np.exp(-dist_test / (2 * length_scale ** 2))

        # 预测 — 用 solve 替代 inv，数值更稳定
        mu = K_star @ np.linalg.solve(K, y_train)

        dist_test_test = np.sum((X_test[:, None] - X_test[None, :]) ** 2, axis=-1)
        K_star_star = np.exp(-dist_test_test / (2 * length_scale ** 2))
        V = np.linalg.solve(K, K_star.T)
        sigma = np.sqrt(np.abs(np.diag(K_star_star - K_star @ V)))
        return mu, sigma

    def _generate_random_config(self) -> Dict[str, str]:
        """随机生成配置"""
        config = {}
        for param in ["role", "format", "reasoning", "fewshot"]:
            options = self.templates.get_options(param)
            config[param] = np.random.choice(options)
        return config

    def _evaluate_config(self, config: Dict[str, str], questions: List[Dict], model_fn) -> float:
        """评估配置得分"""
        scores = []
        for q in questions:
            prompt = self.builder.build(q["question"], config)
            answer = model_fn(prompt)
            score = self.scorer.score(q["question"], answer, q.get("expected_hint", ""))
            scores.append(score)
        return float(np.mean(scores))

    def search(self, questions: List[Dict], model_fn, *,
               n_iterations: int = 20, n_init: int = 5,
               max_questions: Optional[int] = None) -> BayesianSearchResult:
        """
        贝叶斯优化搜索

        Args:
            questions: 评测题目
            model_fn: 模型调用函数
            n_iterations: 贝叶斯迭代次数
            n_init: 随机初始化数量
            max_questions: 最大题目数

        Returns:
            BayesianSearchResult
        """
        if max_questions:
            questions = questions[:max_questions]

        # 1. 随机初始化
        print(f"初始化: 随机评估 {n_init} 个配置...")
        X_train_list = []
        y_train_list = []
        history = []

        for i in range(n_init):
            config = self._generate_random_config()
            score = self._evaluate_config(config, questions, model_fn)
            X_train_list.append(self._config_to_vector(config))
            y_train_list.append(score)
            history.append(max(y_train_list))
            name = self.templates.config_to_name(config)
            print(f"  [{i+1}/{n_init}] {name}: {score:.2f}")

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        best_idx = np.argmax(y_train)
        best_score = y_train[best_idx]
        best_config = self._vector_to_config(X_train[best_idx])

        # 2. 贝叶斯优化循环
        for iteration in range(n_iterations):
            print(f"\n迭代 {iteration+1}/{n_iterations} (当前最优: {best_score:.2f})")

            # 生成候选
            candidates = [self._generate_random_config() for _ in range(200)]
            X_candidates = np.array([self._config_to_vector(c) for c in candidates])

            # GP 预测
            mu, sigma = self._gp_predict(X_train, y_train, X_candidates)

            # 计算 EI
            ei_values = self._expected_improvement(mu, sigma, best_score)

            # 选择最优候选
            best_candidate_idx = np.argmax(ei_values)
            next_config = candidates[best_candidate_idx]

            # 评估
            score = self._evaluate_config(next_config, questions, model_fn)
            X_train = np.vstack([X_train, self._config_to_vector(next_config)])
            y_train = np.append(y_train, score)
            history.append(max(y_train))

            name = self.templates.config_to_name(next_config)
            print(f"  评估: {name}: {score:.2f}")

            if score > best_score:
                best_score = score
                best_config = next_config
                print(f"  ★ 新最优!")

        return BayesianSearchResult(best_config, best_score, X_train, y_train, history)
