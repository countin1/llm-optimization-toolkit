"""
遗传算法搜索

用进化策略搜索 prompt 模板空间。
比网格搜索更高效，比贝叶斯更简单。
"""

from typing import Dict, List, Optional
import numpy as np
from ..core.templates import PromptTemplates
from ..core.builder import PromptBuilder
from ..core.scorer import Scorer


class GeneticSearchResult:
    """遗传算法结果"""

    def __init__(self, best_config: Dict, best_score: float,
                 history: List[float], generations: int):
        self.best_config = best_config
        self.best_score = best_score
        self.history = history
        self.generations = generations

    def summary(self) -> str:
        templates = PromptTemplates()
        name = templates.config_to_name(self.best_config)
        return (
            f"最优配置: {name}\n"
            f"最优得分: {self.best_score:.2f}\n"
            f"迭代代数: {self.generations}\n"
            f"初始最优: {self.history[0]:.2f}\n"
            f"最终最优: {self.history[-1]:.2f}\n"
            f"提升: +{self.history[-1] - self.history[0]:.2f}"
        )


class GeneticSearch:
    """遗传算法搜索"""

    def __init__(self, templates: Optional[PromptTemplates] = None,
                 builder: Optional[PromptBuilder] = None,
                 scorer: Optional[Scorer] = None):
        self.templates = templates or PromptTemplates()
        self.builder = builder or PromptBuilder(self.templates)
        self.scorer = scorer or Scorer()

    def _random_config(self) -> Dict[str, str]:
        """随机生成配置"""
        config = {}
        for param in ["role", "format", "reasoning", "fewshot"]:
            options = self.templates.get_options(param)
            config[param] = np.random.choice(options)
        return config

    def _evaluate(self, config: Dict[str, str], questions: List[Dict], model_fn) -> float:
        """评估配置"""
        scores = []
        for q in questions:
            prompt = self.builder.build(q["question"], config)
            answer = model_fn(prompt)
            score = self.scorer.score(q["question"], answer, q.get("expected_hint", ""))
            scores.append(score)
        return float(np.mean(scores))

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """交叉"""
        child = {}
        for param in ["role", "format", "reasoning", "fewshot"]:
            child[param] = parent1[param] if np.random.random() < 0.5 else parent2[param]
        return child

    def _mutate(self, config: Dict, mutation_rate: float = 0.2) -> Dict:
        """变异"""
        mutated = config.copy()
        for param in ["role", "format", "reasoning", "fewshot"]:
            if np.random.random() < mutation_rate:
                options = self.templates.get_options(param)
                mutated[param] = np.random.choice(options)
        return mutated

    def search(self, questions: List[Dict], model_fn, *,
               population_size: int = 10, generations: int = 10,
               mutation_rate: float = 0.2, elite_ratio: float = 0.3,
               max_questions: Optional[int] = None) -> GeneticSearchResult:
        """
        遗传算法搜索

        Args:
            questions: 评测题目
            model_fn: 模型调用函数
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            elite_ratio: 精英保留比例
            max_questions: 最大题目数

        Returns:
            GeneticSearchResult
        """
        if max_questions:
            questions = questions[:max_questions]

        n_elite = max(1, int(population_size * elite_ratio))
        history = []

        # 初始化种群
        print(f"初始化种群: {population_size} 个个体")
        population = [self._random_config() for _ in range(population_size)]
        fitness = [self._evaluate(ind, questions, model_fn) for ind in population]

        best_idx = int(np.argmax(fitness))
        best_score = fitness[best_idx]
        best_config = population[best_idx]
        history.append(best_score)

        print(f"  初始最优: {best_score:.2f}")

        # 进化循环
        for gen in range(generations):
            # 选择精英
            sorted_indices = np.argsort(fitness)[::-1]
            elites = [population[i] for i in sorted_indices[:n_elite]]

            # 生成下一代
            new_population = elites.copy()  # 保留精英
            while len(new_population) < population_size:
                # 选择父母（锦标赛选择）
                i1, i2 = np.random.choice(len(elites), 2, replace=False)
                parent1 = elites[i1]
                parent2 = elites[i2]

                # 交叉
                child = self._crossover(parent1, parent2)

                # 变异
                child = self._mutate(child, mutation_rate)

                new_population.append(child)

            # 评估新种群
            population = new_population
            fitness = [self._evaluate(ind, questions, model_fn) for ind in population]

            gen_best_idx = int(np.argmax(fitness))
            gen_best_score = fitness[gen_best_idx]
            history.append(max(history[-1], gen_best_score))

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_config = population[gen_best_idx]
                print(f"  代 {gen+1}: {gen_best_score:.2f} ★ 新最优!")
            else:
                print(f"  代 {gen+1}: {gen_best_score:.2f}")

        return GeneticSearchResult(best_config, best_score, history, generations)
