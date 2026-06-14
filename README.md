[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![CI](https://github.com/countin1/llm-optimization-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/countin1/llm-optimization-toolkit/actions/workflows/ci.yml)

# 🚀 LLM Optimization Toolkit

> 大模型评测 + 优化工具包，包含 Prompt 自动优化和 LoRA 微调两大项目。

## 📦 包含项目

### 🔥 [PromptForge](./promptforge/) — Prompt 自动优化框架

类 DSPy 的 Prompt 模板搜索引擎，给定评测集自动找到最优 prompt 结构。

**核心功能：**
- 网格搜索：遍历所有模板组合
- 贝叶斯优化：高斯过程 + 期望改进
- 遗传算法：进化策略搜索
- 组件分析：逐个组件测试贡献
- 统计检验：配对 t 检验 + Cohen's d
- 可视化：模板对比图、收敛曲线、热力图
- 高级统计：Bootstrap CI、交叉验证、功效分析

**面试话术：**
> "我用网格搜索 + 贝叶斯优化在 36 种模板中找到最优组合，统计题提升 15%，Cohen's d = 0.73，p < 0.01。"

---

### 🔧 [LoRAForge](./loraforge/) — 大模型微调 Pipeline

端到端的 LoRA/QLoRA 微调 + 评测 pipeline。

**核心功能：**
- 数据准备：加载、增强、分层划分
- LoRA 微调：支持 Qwen/LLaMA
- QLoRA：4bit 量化，省显存 70%
- 超参搜索：r/alpha/lr/epochs 网格搜索
- 训练监控：loss 曲线、checkpoint 管理
- 统计验证：配对 t 检验 + Cohen's d

**面试话术：**
> "我用 LoRA 微调 Qwen2.5-7B，统计知识维度 Cohen's d = 0.73，p < 0.01。QLoRA 4bit 量化让 7B 模型只需 6GB 显存。"

---

## 🏗️ 工程化

- ✅ Docker 容器化
- ✅ GitHub Actions CI/CD
- ✅ 单元测试覆盖
- ✅ 类型提示
- ✅ 模块化设计

## 📊 统计方法

| 方法 | 用途 |
|------|------|
| 配对 t 检验 | 两组得分差异显著性 |
| Cohen's d | 效应量（大/中/小） |
| Bootstrap CI | 置信区间估计 |
| 功效分析 | 所需样本量计算 |
| Bonferroni 校正 | 多重比较校正 |
| 交叉验证 | 泛化能力评估 |

## 📚 依赖

- Python >= 3.10
- numpy, scipy, matplotlib
- openai (PromptForge)
- torch, transformers, peft (LoRAForge)

## 📄 License

MIT
