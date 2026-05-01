"""
Agent RL Demo: rllm + TRL integration on Mac Air

Usage:
    # Natural language:
    python -m rllm_trl.train "用 qwen-0.5b 训练数学 agent，64 个问题，2 个 epoch"
    python -m rllm_trl.train "quick test with 16 problems"

    # Default config:
    python -m rllm_trl.train
"""

import os
import re
import sys
import warnings

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from datasets import Dataset
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, PrinterCallback
from trl import GRPOConfig, GRPOTrainer

from rllm_trl.base import ToolOutput
from rllm_trl.config import TrainingConfig, parse_natural_language
from rllm_trl.logger import TrainingLogger
from rllm_trl.math_env import MathCalcEnv, generate_math_problems
from rllm_trl.perf_stats import PerfTracker
from rllm_trl.rollout import make_rllm_rollout_func
from rllm_trl.tool_agent import ToolAgent
from rllm_trl.trajectory_writer import TrajectoryWriter


def build_dataset(problems):
    records = []
    for p in problems:
        records.append({
            "prompt": [{"role": "user", "content": p["question"]}],
            "answer": p["answer"],
        })
    return Dataset.from_list(records)


TOOL_SYSTEM_PROMPT = """You are a helpful math assistant. You can use the calculate tool to compute arithmetic expressions, and the finish tool to give your final answer.

Always use the calculate tool first, then use finish to report the result."""


class CalculateTool:
    name = "calculate"
    description = "Evaluate a mathematical expression"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 + 3'"
            }
        },
        "required": ["expression"]
    }

    def __init__(self, **kwargs):
        pass

    def __call__(self, expression: str = "", **kwargs) -> ToolOutput:
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in str(expression)):
                return ToolOutput(output="Error: invalid expression")
            result = eval(str(expression))  # noqa: S307
            return ToolOutput(output=str(result))
        except Exception as e:
            return ToolOutput(output=f"Error: {e}")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class FinishTool:
    name = "finish"
    description = "Submit your final answer"
    parameters = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Your final answer"
            }
        },
        "required": ["response"]
    }

    def __init__(self, **kwargs):
        pass

    def __call__(self, response: str = "", **kwargs) -> ToolOutput:
        return ToolOutput(output=response)

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


def create_simple_tool_agent():
    return {
        "system_prompt": TOOL_SYSTEM_PROMPT,
        "parser_name": "qwen",
        "tool_map": {
            "calculate": CalculateTool,
            "finish": FinishTool,
        },
    }


class RllmCallback(TrainerCallback):
    def __init__(self, training_logger):
        self.training_logger = training_logger
        self._steps_synced = False

    def on_step_begin(self, args, state, control, **kwargs):
        if not self._steps_synced and state.max_steps > 0:
            self.training_logger.total_steps = state.max_steps
            self._steps_synced = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.training_logger.update_training_metrics(logs)


def math_reward_fn(prompts, completions, **kwargs):
    rewards = []
    answers = kwargs.get("answer", [])
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        gt = answers[i] if i < len(answers) else None
        if isinstance(completion, list):
            text = " ".join(
                msg.get("content", "") for msg in completion if isinstance(msg, dict)
            )
        else:
            text = str(completion)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers and gt is not None:
            try:
                predicted = float(numbers[-1])
                expected = float(gt)
                rewards.append(1.0 if abs(predicted - expected) < 1e-6 else 0.0)
            except (ValueError, TypeError):
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def main(config: TrainingConfig | None = None):
    if config is None:
        config = TrainingConfig()

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config.to_json(os.path.join(output_dir, "config.json"))

    log_file = os.path.join(output_dir, "training_log.txt")
    log = TrainingLogger(verbose=config.verbose, log_file=log_file)
    log.log_training_start(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=dtype, trust_remote_code=True,
    )
    param_count = sum(p.numel() for p in model.parameters())
    log.log_model_loaded(config.model_name, device, param_count)

    problems = generate_math_problems(n=config.num_problems, seed=config.seed, difficulty=config.difficulty)
    dataset = build_dataset(problems)
    log.log_dataset_ready(len(dataset), dataset[0])

    traj_writer = TrajectoryWriter(output_dir, enabled=True)
    perf = PerfTracker(output_dir, enabled=True)

    agent_args = create_simple_tool_agent()

    answer_map = {p["question"]: p["answer"] for p in problems}

    rollout_func = make_rllm_rollout_func(
        agent_class=ToolAgent,
        agent_args=agent_args,
        env_class=MathCalcEnv,
        env_args={"max_steps": config.max_agent_steps},
        max_steps=config.max_agent_steps,
        max_response_length=config.max_response_length,
        max_prompt_length=config.max_prompt_length,
        sampling_params={"temperature": config.temperature, "top_p": config.top_p},
        training_logger=log,
        trajectory_writer=traj_writer,
        perf_tracker=perf,
        answer_map=answer_map,
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        log_level="error",
        save_strategy="no",
        bf16=False,
        fp16=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
    )

    log.log_trainer_ready(config.num_epochs)

    perf.start_training()

    logging.getLogger("trl").setLevel(logging.ERROR)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[math_reward_fn],
        rollout_func=rollout_func,
        callbacks=[RllmCallback(log)],
    )

    trainer.remove_callback(PrinterCallback)

    trainer.train()

    # Sync perf stats into logger for the report
    log.set_perf_stats(perf.rollout_stats)
    perf.save_to_file()

    perf_summary = perf.get_summary()
    log.print_training_report(config, perf_summary, output_dir)

    if config.save_model:
        save_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log.log_model_saved(save_path)

    log.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = " ".join(sys.argv[1:])
        if arg.endswith(".json") and os.path.isfile(arg):
            cfg = TrainingConfig.from_json(arg)
        else:
            cfg = parse_natural_language(arg)
    else:
        cfg = TrainingConfig()
    main(cfg)
