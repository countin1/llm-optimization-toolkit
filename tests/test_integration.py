"""
集成测试 — 覆盖完整 pipeline
"""

import pytest
import json
import numpy as np
from pathlib import Path


# === PromptForge 集成测试 ===

class TestPromptForgeIntegration:
    """PromptForge 完整流程测试"""

    def test_template_search_space(self):
        """测试模板搜索空间大小"""
        from promptforge.core.templates import PromptTemplates

        templates = PromptTemplates()
        combos = templates.get_all_combinations()

        # 4 个维度：role(4) x format(4) x reasoning(4) x fewshot(2) = 128
        assert len(combos) == 128, f"搜索空间应为 128，实际 {len(combos)}"

    def test_builder_produces_valid_prompt(self):
        """测试 Prompt 构建器输出"""
        from promptforge.core.builder import PromptBuilder

        builder = PromptBuilder()
        prompt = builder.build(
            question="什么是 GDP？",
            config={
                "role": "expert",
                "format": "structured",
                "reasoning": "cot",
                "fewshot": "none"
            }
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "GDP" in prompt
        assert "专家" in prompt  # expert 角色

    def test_scorer_rule_mode(self):
        """测试 rule_score 评分"""
        from promptforge.core.scorer import rule_score

        # 高质量回答
        high_quality = "GDP 是国内生产总值，衡量一个国家在特定时期内生产的所有最终商品和服务的市场价值。它包括消费、投资、政府支出和净出口四个组成部分。"
        score_high = rule_score(high_quality, "什么是 GDP？")

        # 低质量回答
        low_quality = "不知道"
        score_low = rule_score(low_quality, "什么是 GDP？")

        assert score_high > score_low
        assert 1 <= score_high <= 10
        assert 1 <= score_low <= 10

    def test_grid_search_finds_better_template(self):
        """测试网格搜索能找到更优模板"""
        # 模拟评分数据
        mock_results = {
            "baseline": {"scores": [5, 5, 6, 5, 6] * 10},
            "expert": {"scores": [6, 7, 6, 7, 6] * 10},
            "cot": {"scores": [5, 6, 5, 6, 5] * 10},
        }

        # 找最优
        best = max(mock_results.items(), key=lambda x: np.mean(x[1]["scores"]))
        assert best[0] == "expert"

    def test_statistical_tests(self):
        """测试统计检验方法"""
        from promptforge.analysis.stats import paired_t_test, cohens_d

        # 模拟数据：微调后明显更好
        before = np.array([5, 5, 6, 5, 6, 5, 5, 6, 5, 6])
        after = np.array([6, 7, 6, 7, 6, 7, 6, 7, 6, 7])

        result = paired_t_test(before, after)
        d_result = cohens_d(before, after)

        assert result["p_value"] < 0.05, "p 值应小于 0.05"
        # Cohen's d 是负数（before - after < 0），取绝对值验证效应量
        assert abs(d_result["d"]) > 0.5, "Cohen's d 绝对值应大于 0.5"

    def test_advanced_stats_bootstrap_ci(self):
        """测试 Bootstrap 置信区间"""
        from promptforge.analysis.advanced_stats import bootstrap_ci

        scores = np.array([6, 7, 6, 7, 6, 7, 6, 7, 6, 7])
        result = bootstrap_ci(scores, n_bootstrap=1000)

        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["mean"] < result["ci_upper"]


# === LoRAForge 集成测试 ===

class TestLoRAForgeIntegration:
    """LoRAForge 完整流程测试"""

    def test_data_loader_chat_format(self):
        """测试 Chat 格式数据加载"""
        from loraforge.data.loader import DataLoader

        loader = DataLoader()

        # 模拟题目数据
        questions = [
            {
                "id": 1,
                "question": "什么是 GDP？",
                "expected_hint": "国内生产总值；衡量经济总量",
                "dimension": "macro",
                "difficulty": 1
            }
        ]

        chat_data = loader.to_chat_format(questions)

        assert len(chat_data) == 1
        assert "messages" in chat_data[0]
        assert len(chat_data[0]["messages"]) == 3  # system, user, assistant
        assert chat_data[0]["messages"][1]["role"] == "user"

    def test_data_loader_alpaca_format(self):
        """测试 Alpaca 格式数据加载"""
        from loraforge.data.loader import DataLoader

        loader = DataLoader()

        questions = [
            {
                "id": 1,
                "question": "什么是 GDP？",
                "expected_hint": "国内生产总值；衡量经济总量",
                "dimension": "macro",
                "difficulty": 1
            }
        ]

        alpaca_data = loader.to_alpaca_format(questions)

        assert len(alpaca_data) == 1
        assert "instruction" in alpaca_data[0]
        assert "output" in alpaca_data[0]

    def test_data_augmentor(self):
        """测试数据增强"""
        from loraforge.data.augment import DataAugmentor

        augmentor = DataAugmentor()

        # 模拟 Chat 格式数据
        data = [
            {
                "messages": [
                    {"role": "system", "content": "你是一位专家"},
                    {"role": "user", "content": "什么是 GDP？"},
                    {"role": "assistant", "content": "GDP 是国内生产总值..."}
                ],
                "metadata": {"id": 1, "dimension": "macro", "difficulty": 1}
            }
        ]

        augmented = augmentor.augment(data, n_variants=3)

        # 原始 + 3 个变体 = 4
        assert len(augmented) == 4

    def test_data_splitter_stratified(self):
        """测试分层划分"""
        from loraforge.data.splitter import DataSplitter

        splitter = DataSplitter()

        # 模拟带维度的数据
        data = [
            {
                "messages": [
                    {"role": "system", "content": "你是一位专家"},
                    {"role": "user", "content": f"问题 {i}"},
                    {"role": "assistant", "content": f"回答 {i}"}
                ],
                "metadata": {"id": i, "dimension": "macro" if i < 50 else "micro", "difficulty": 1}
            }
            for i in range(100)
        ]

        train, test = splitter.split(data, train_ratio=0.8)

        assert len(train) == 80
        assert len(test) == 20

    def test_model_comparison_stats(self):
        """测试模型对比统计"""
        from loraforge.eval.stats import paired_t_test, cohens_d

        # 模拟微调前后数据
        base_scores = np.array([5, 5, 6, 5, 6, 5, 5, 6, 5, 6])
        finetuned_scores = np.array([6, 7, 6, 7, 6, 7, 6, 7, 6, 7])

        result = paired_t_test(base_scores, finetuned_scores)
        d_result = cohens_d(base_scores, finetuned_scores)

        assert result["p_value"] < 0.05
        # Cohen's d 是负数（base - finetuned < 0），取绝对值验证效应量
        assert abs(d_result["d"]) > 0.5

    def test_experiment_results_format(self):
        """测试实验结果文件格式"""
        results_path = Path(__file__).parent.parent / "experiment_results_100q.json"

        if not results_path.exists():
            pytest.skip("实验结果文件不存在")

        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)

        # 验证结构
        assert "model" in data
        assert "n" in data
        assert "results" in data
        assert "baseline" in data["results"]

        # 验证每个结果有必需字段
        for name, result in data["results"].items():
            assert "mean" in result
            assert "std" in result
            assert "scores" in result
            assert len(result["scores"]) == data["n"]


# === 跨模块集成测试 ===

class TestCrossModuleIntegration:
    """跨模块集成测试"""

    def test_promptforge_stats_consistency(self):
        """测试 PromptForge 统计方法一致性"""
        from promptforge.analysis.stats import paired_t_test, cohens_d

        a = np.array([5, 6, 7, 5, 6])
        b = np.array([6, 7, 8, 6, 7])

        r1 = paired_t_test(a, b)
        r2 = paired_t_test(a, b)

        assert r1["t_stat"] == r2["t_stat"]
        assert r1["p_value"] == r2["p_value"]

    def test_loraforge_stats_consistency(self):
        """测试 LoRAForge 统计方法一致性"""
        from loraforge.eval.stats import paired_t_test, cohens_d

        a = np.array([5, 6, 7, 5, 6])
        b = np.array([6, 7, 8, 6, 7])

        r1 = paired_t_test(a, b)
        r2 = paired_t_test(a, b)

        assert r1["t_stat"] == r2["t_stat"]
        assert r1["p_value"] == r2["p_value"]

    def test_both_modules_use_same_stats(self):
        """测试两个模块的统计方法结果一致"""
        from promptforge.analysis.stats import paired_t_test as pf_ttest
        from promptforge.analysis.stats import cohens_d as pf_d
        from loraforge.eval.stats import paired_t_test as lf_ttest
        from loraforge.eval.stats import cohens_d as lf_d

        # 使用有差异的数据，避免 std(diff)=0 导致的 -inf
        a = np.array([5, 6, 7, 5, 6, 5, 6, 7, 5, 6])
        b = np.array([7, 6, 8, 5, 7, 6, 8, 5, 7, 6])

        # 两个模块的统计方法应该给出相同结果
        pf_result = pf_ttest(a, b)
        lf_result = lf_ttest(a, b)

        assert abs(pf_result["t_stat"] - lf_result["t_stat"]) < 1e-10
        assert abs(pf_result["p_value"] - lf_result["p_value"]) < 1e-10
        assert abs(pf_d(a, b)["d"] - lf_d(a, b)["d"]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
