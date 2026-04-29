"""
Training configuration and natural language launcher.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Dataset
    num_problems: int = 64
    seed: int = 42

    # Agent / Environment
    task_type: str = "math"
    max_agent_steps: int = 3
    max_response_length: int = 256
    max_prompt_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Training
    num_epochs: int = 2
    batch_size: int = 2
    num_generations: int = 4
    max_completion_length: int = 256
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 2

    # Output
    run_id: str = ""
    output_dir: str = ""
    save_model: bool = True

    # Logging
    logging_steps: int = 1
    verbose: bool = True

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"run_{int(time.time())}"
        if not self.output_dir:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "runs")
            self.output_dir = os.path.join(base, self.run_id)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Training Configuration",
            "=" * 60,
            f"  Run ID:           {self.run_id}",
            f"  Task type:        {self.task_type}",
            f"  Model:            {self.model_name}",
            f"  Problems:         {self.num_problems} (seed={self.seed})",
            f"  Agent steps:      {self.max_agent_steps}",
            f"  Temperature:      {self.temperature}",
            f"  Epochs:           {self.num_epochs}",
            f"  Batch size:       {self.batch_size}",
            f"  Generations/prompt: {self.num_generations}",
            f"  Learning rate:    {self.learning_rate}",
            f"  Grad accum steps: {self.gradient_accumulation_steps}",
            f"  Output:           {self.output_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_json(self, path: str | None = None) -> str:
        data = asdict(self)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = json.load(f)
        known_fields = {f.name for f in __import__("dataclasses").fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# Keyword mappings for natural language parsing
MODEL_ALIASES = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}


def parse_natural_language(description: str) -> TrainingConfig:
    """Parse a natural language training description into a TrainingConfig.

    Examples:
        "用 qwen-0.5b 训练数学 agent，100 个问题，3 个 epoch"
        "train math agent with qwen-1.5b, 200 problems, lr=5e-6"
        "快速测试，16 个问题，1 epoch"
    """
    import re
    config = TrainingConfig()
    desc = description.lower()

    # Model
    for alias, full_name in MODEL_ALIASES.items():
        if alias in desc:
            config.model_name = full_name
            break

    # Number of problems
    m = re.search(r'(\d+)\s*(?:个|道)?(?:问题|题目|problems?|samples?|examples?)', desc)
    if m:
        config.num_problems = int(m.group(1))

    # Epochs
    m = re.search(r'(\d+)\s*(?:个)?\s*(?:epoch|轮)', desc)
    if m:
        config.num_epochs = int(m.group(1))

    # Learning rate
    m = re.search(r'lr\s*=?\s*([\d.e-]+)', desc)
    if m:
        config.learning_rate = float(m.group(1))

    # Batch size
    m = re.search(r'batch\s*(?:_?size)?\s*=?\s*(\d+)', desc)
    if m:
        config.batch_size = int(m.group(1))

    # Temperature
    m = re.search(r'(?:temp|temperature)\s*=?\s*([\d.]+)', desc)
    if m:
        config.temperature = float(m.group(1))

    # Generations
    m = re.search(r'(\d+)\s*(?:个)?(?:generations?|生成)', desc)
    if m:
        config.num_generations = int(m.group(1))

    # Max steps
    m = re.search(r'(\d+)\s*(?:个)?(?:steps?|步)', desc)
    if m:
        config.max_agent_steps = int(m.group(1))

    # Quick test mode
    if any(kw in desc for kw in ("快速测试", "quick test", "fast test", "测试一下")):
        config.num_problems = 16
        config.num_epochs = 1
        config.batch_size = 2
        config.num_generations = 2

    # Task type
    if any(kw in desc for kw in ("数学", "math", "计算", "calc")):
        config.task_type = "math"
    elif any(kw in desc for kw in ("代码", "code", "coding")):
        config.task_type = "code"
    elif any(kw in desc for kw in ("搜索", "search")):
        config.task_type = "search"

    return config
