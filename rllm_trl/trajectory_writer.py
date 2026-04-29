"""
Trajectory file writer — saves rollout trajectories to JSONL for inspection.
Each line is one trajectory with full conversation, metrics, and reward.
"""

import json
import os
import time
from pathlib import Path


class TrajectoryWriter:
    def __init__(self, output_dir, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        self.traj_dir = os.path.join(output_dir, "trajectories")
        os.makedirs(self.traj_dir, exist_ok=True)
        self._global_traj_id = 0

    def write_rollout(self, step, agents, envs, results, tokenizer=None):
        if not self.enabled:
            return
        filepath = os.path.join(self.traj_dir, f"step_{step:04d}.jsonl")
        with open(filepath, "w") as f:
            for i, (agent, env, result) in enumerate(zip(agents, envs, results)):
                self._global_traj_id += 1
                record = self._build_record(
                    traj_id=self._global_traj_id,
                    step=step,
                    index=i,
                    agent=agent,
                    env=env,
                    result=result,
                    tokenizer=tokenizer,
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_record(self, traj_id, step, index, agent, env, result, tokenizer):
        record = {
            "trajectory_id": traj_id,
            "step": step,
            "index": index,
            "question": getattr(env, "question", ""),
            "expected_answer": getattr(env, "answer", ""),
            "chat_completions": agent.chat_completions if hasattr(agent, "chat_completions") else [],
        }

        if isinstance(result, dict):
            metrics = result.get("metrics", {})
            record["reward"] = result.get("trajectory_reward", 0.0)
            record["num_steps"] = metrics.get("steps", 0)
            record["prompt_tokens"] = len(result.get("prompt_tokens", []))
            record["response_tokens"] = len(result.get("response_tokens", []))
            record["response_masks_sum"] = int(sum(
                result.get("response_masks", [])
                if not hasattr(result.get("response_masks", []), "sum")
                else [result["response_masks"].sum().item()]
            ))
            record["timing"] = {
                "llm_time": metrics.get("llm_time", 0),
                "env_time": metrics.get("env_time", 0),
                "total_time": metrics.get("total_time", 0),
                "reward_time": metrics.get("reward_time"),
            }

            if tokenizer and "response_tokens" in result:
                tokens = result["response_tokens"]
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                record["response_text"] = tokenizer.decode(tokens, skip_special_tokens=True)

        return record

    def write_summary(self, step, summary_stats):
        if not self.enabled:
            return
        filepath = os.path.join(self.traj_dir, f"step_{step:04d}_summary.json")
        with open(filepath, "w") as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
