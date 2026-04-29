# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Agent RL training project that integrates rLLM concepts with HuggingFace TRL. The repo contains three directories:

- `rllm_trl/` — The active codebase. A self-contained agent RL training pipeline that combines rLLM's agent/environment abstractions with TRL's GRPOTrainer. Deliberately inlines minimal rllm abstractions to avoid rllm's heavy dependency chain (vllm, flash-attn, deepspeed). Runs on Mac (MPS) and CPU.
- `rllm/` — Upstream rLLM framework (reference only, archived). Full RL training framework for language agents using verl, vllm, deepspeed. Not directly used by rllm_trl.
- `trl/` — Upstream TRL library (HuggingFace). Provides GRPOTrainer, SFTTrainer, and other RL trainers. Used as a dependency by rllm_trl.

## Running Training

```bash
# Default config (Qwen2.5-0.5B, 64 problems, 2 epochs)
python -m rllm_trl.train

# Natural language config (supports Chinese and English)
python -m rllm_trl.train "用 qwen-0.5b 训练数学 agent，64 个问题，2 个 epoch"
python -m rllm_trl.train "quick test with 16 problems"
```

Training outputs go to `rllm_trl/output/`: trajectories (JSONL), perf_stats.json, final_model/.

## rllm_trl Architecture

The training pipeline flows: `train.py` → `GRPOTrainer` → `rollout_func` → `HFAgentExecutionEngine` → agent/env loop.

Key modules and their roles:

- `train.py` — Entry point. Builds dataset, model, tokenizer, wires everything into GRPOTrainer with a custom rollout function. Contains the math reward function and tool definitions (CalculateTool, FinishTool).
- `config.py` — `TrainingConfig` dataclass with all hyperparameters. `parse_natural_language()` converts free-text descriptions (Chinese/English) into config via regex matching. Model aliases map shorthand like "qwen-0.5b" to full HF paths.
- `rollout.py` — `make_rllm_rollout_func()` returns a closure that GRPOTrainer calls each step. Creates an HFAgentExecutionEngine, runs async trajectories, computes logprobs, returns prompt/completion/mask tensors. This is the bridge between TRL and rllm-style agent execution.
- `hf_engine.py` — `HFAgentExecutionEngine` runs agent-environment loops using HuggingFace model.generate() for inference. Manages async parallel trajectories, token-level prompt/response splitting with masks (1=model, 0=env), and MC return computation.
- `base.py` — Core abstractions inlined from rllm: `BaseAgent`, `BaseEnv`, `Step`, `Action`, `Trajectory`, `ToolCall`, `ToolOutput`. BaseEnv follows gym-style reset()/step() interface.
- `tool_agent.py` — `ToolAgent(BaseAgent)` manages conversation history, parses tool calls from model output via QwenToolParser, formats observations as messages. Falls back to "finish" tool when no tool call is parsed.
- `math_env.py` — `MathCalcEnv(BaseEnv)` implements a calculator environment. `generate_math_problems()` creates arithmetic datasets. Reward is binary: 1.0 if answer matches, 0.0 otherwise.
- `parsers.py` — Chat template parsers (generic + Qwen-specific) and tool call parsers. `convert_messages_to_tokens_and_masks()` is critical: it tokenizes messages and produces per-token masks distinguishing model-generated vs environment tokens for GRPO training.
- `logger.py` — `TrainingLogger` prints a live progress table during training and a summary report at the end with reward trends, training dynamics, and performance breakdown.
- `perf_stats.py` — `PerfTracker` tracks timing at rollout/step/token level. Breaks down wall time into LLM inference, env execution, logprob computation, and GRPO training.
- `trajectory_writer.py` — Saves per-step JSONL files with full conversation, metrics, and decoded response text.

## Key Design Decisions

The response mask system (1=model tokens, 0=env tokens) is central to correct GRPO training — only model-generated tokens should receive gradient updates. This masking happens in `hf_engine.py` via `convert_messages_to_tokens_and_masks()`.

The rollout function handles asyncio event loop edge cases (running loop detection, thread pool fallback) because TRL's training loop may already have an active event loop.

## TRL Codebase Conventions

Per trl/CLAUDE.md: trainers are self-contained by design with deliberately duplicated logic. When modifying duplicated code across trainers (e.g., vLLM generation paths), apply the same change to all trainers. Consistency across copies is mandatory — even over correctness.

## Linting

rllm uses ruff with relaxed line-length (5000). Key ignored rules: F405/F403 (star imports), E731 (lambda assignment), B007 (unused loop var). `rllm/rewards/code_utils/` is excluded from all lint rules.
