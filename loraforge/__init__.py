"""
LoRAForge — 大模型微调 Pipeline

端到端的 LoRA/QLoRA 微调 + 评测 pipeline，从数据准备到统计验证。

功能：
- 数据准备：加载、增强、分层划分
- LoRA 微调：支持 Qwen/LLaMA 等模型
- QLoRA：4bit 量化，省显存 70%
- 超参搜索：r/alpha/lr/epochs 网格搜索
- 统计验证：配对 t 检验 + Cohen's d

用法：
    from loraforge import LoRAForge

    forge = LoRAForge(model_name="Qwen/Qwen2.5-7B")
    forge.prepare_data(augment=3)
    forge.train(epochs=3, qlora=True)
    result = forge.evaluate()
    print(result.summary())
"""

__version__ = "1.0.0"
__author__ = "countin1"

from .data.loader import DataLoader
from .data.augment import DataAugmentor
from .data.splitter import DataSplitter
from .train.lora import LoRATrainer
from .train.qlora import QLoRATrainer
from .eval.compare import ModelComparator
from .eval.stats import paired_t_test, cohens_d
