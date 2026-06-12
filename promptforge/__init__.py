"""
PromptForge — Prompt 自动优化框架

类 DSPy 的 Prompt 模板搜索引擎，给定评测集自动找到最优 prompt 结构。

功能：
- 网格搜索：遍历所有模板组合
- 贝叶斯优化：高斯过程 + 期望改进
- 遗传算法：进化策略搜索
- 组件分析：逐个组件测试贡献
- 统计检验：配对 t 检验 + Cohen's d

用法：
    from promptforge import PromptForge

    forge = PromptForge(config_path="config.yaml")
    result = forge.search(method="bayesian", iterations=20)
    print(result.best_config, result.best_score)
"""

__version__ = "1.0.0"
__author__ = "countin1"

from .core.templates import PromptTemplates
from .core.builder import PromptBuilder
from .core.scorer import Scorer
from .search.grid import GridSearch
from .search.bayesian import BayesianSearch
from .search.genetic import GeneticSearch
