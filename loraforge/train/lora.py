"""
LoRA 微调

使用 PEFT 库进行 LoRA 微调。
"""

import os
from typing import Dict, List, Optional


class LoRATrainer:
    """LoRA 微调器"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B",
                 r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 target_modules: Optional[List[str]] = None):
        self.model_name = model_name
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    def train(self, train_data: List[Dict], test_data: List[Dict],
              epochs: int = 3, batch_size: int = 4, lr: float = 2e-4,
              output_dir: str = "outputs/adapters") -> Dict:
        """
        运行 LoRA 微调

        Args:
            train_data: 训练数据
            test_data: 测试数据
            epochs: 训练轮次
            batch_size: 批大小
            lr: 学习率
            output_dir: 输出目录

        Returns:
            训练结果
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset

        print(f"\n{'='*60}")
        print(f"LoRA 微调: {self.model_name}")
        print(f"{'='*60}")
        print(f"训练集: {len(train_data)} 条")
        print(f"测试集: {len(test_data)} 条")
        print(f"轮次: {epochs}")
        print(f"LoRA r={self.r}, alpha={self.lora_alpha}")

        # 加载 tokenizer
        print("\n加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto"
        )

        # LoRA 配置
        print("配置 LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 准备数据
        print("准备训练数据...")
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)

        def tokenize_fn(examples):
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)
            result = tokenizer(texts, truncation=True, max_length=1024, padding="max_length")
            result["labels"] = result["input_ids"].copy()
            return result

        train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
        test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

        # 训练参数
        os.makedirs(output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=10,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True,
            report_to="none",
        )

        # 训练
        print("\n开始训练...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        train_result = trainer.train()

        # 保存
        print("\n保存模型...")
        save_path = os.path.join(output_dir, "lora_adapter")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"\n训练完成!")
        print(f"  训练损失: {train_result.training_loss:.4f}")
        print(f"  训练步数: {train_result.global_step}")
        print(f"  模型保存: {save_path}")

        return {
            "train_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "save_path": save_path,
        }
