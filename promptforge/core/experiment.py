"""
实验追踪器

记录每次实验的配置、结果、耗时，支持对比和复现。
"""

import os
import json
import datetime
from typing import Dict, List, Optional


class Experiment:
    """单次实验"""

    def __init__(self, name: str, config: Dict, tags: List[str] = None):
        self.name = name
        self.config = config
        self.tags = tags or []
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.results = {}
        self.metrics = {}

    def log_metric(self, key: str, value: float):
        """记录指标"""
        self.metrics[key] = value

    def log_result(self, key: str, value):
        """记录结果"""
        self.results[key] = value

    def finish(self):
        """完成实验"""
        self.end_time = datetime.datetime.now()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "config": self.config,
            "tags": self.tags,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "results": self.results,
        }


class ExperimentTracker:
    """实验追踪器"""

    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = output_dir
        self.db_path = os.path.join(output_dir, "experiments.jsonl")
        os.makedirs(output_dir, exist_ok=True)

    def start(self, name: str, config: Dict, tags: List[str] = None) -> Experiment:
        """开始新实验"""
        exp = Experiment(name, config, tags)
        return exp

    def save(self, exp: Experiment):
        """保存实验"""
        exp.finish()
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(exp.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> List[Dict]:
        """加载所有实验"""
        if not os.path.exists(self.db_path):
            return []
        experiments = []
        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    experiments.append(json.loads(line))
        return experiments

    def compare(self, names: List[str]) -> Dict:
        """对比实验"""
        all_exps = self.load_all()
        selected = [e for e in all_exps if e["name"] in names]

        if not selected:
            return {"error": "未找到实验"}

        comparison = {}
        for exp in selected:
            comparison[exp["name"]] = {
                "config": exp["config"],
                "metrics": exp["metrics"],
                "duration": exp["end_time"],
            }

        return comparison

    def get_best(self, metric: str, top_k: int = 5) -> List[Dict]:
        """获取最优实验"""
        all_exps = self.load_all()
        scored = []
        for exp in all_exps:
            if metric in exp.get("metrics", {}):
                scored.append({
                    "name": exp["name"],
                    "score": exp["metrics"][metric],
                    "config": exp["config"],
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
