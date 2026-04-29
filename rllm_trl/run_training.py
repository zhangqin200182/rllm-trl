"""
JSON config-driven training entry point.

Usage:
    python -m rllm_trl.run_training config.json
    python -m rllm_trl.run_training '{"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "num_problems": 32}'
"""

import json
import os
import sys

from rllm_trl.config import TrainingConfig
from rllm_trl.train import main


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m rllm_trl.run_training <config.json | json_string>")
        sys.exit(1)

    arg = sys.argv[1]

    if os.path.isfile(arg):
        config = TrainingConfig.from_json(arg)
    else:
        try:
            data = json.loads(arg)
            known_fields = {f.name for f in __import__("dataclasses").fields(TrainingConfig)}
            filtered = {k: v for k, v in data.items() if k in known_fields}
            config = TrainingConfig(**filtered)
        except json.JSONDecodeError:
            print(f"Error: '{arg}' is not a valid JSON file or JSON string")
            sys.exit(1)

    main(config)
