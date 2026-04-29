"""
Performance statistics for training pipeline.
Tracks timing at rollout, step, and token level.
"""

import json
import os
import time
from dataclasses import dataclass, field


@dataclass
class RolloutStats:
    step: int = 0
    num_trajectories: int = 0
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    total_model_tokens: int = 0
    total_env_tokens: int = 0
    rollout_time: float = 0.0
    llm_time: float = 0.0
    env_time: float = 0.0
    logprob_time: float = 0.0
    rewards: list = field(default_factory=list)

    @property
    def avg_reward(self):
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def tokens_per_second(self):
        if self.llm_time <= 0:
            return 0.0
        return self.total_model_tokens / self.llm_time

    @property
    def avg_response_len(self):
        return self.total_response_tokens / self.num_trajectories if self.num_trajectories else 0

    def to_dict(self):
        return {
            "step": self.step,
            "num_trajectories": self.num_trajectories,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
            "total_model_tokens": self.total_model_tokens,
            "total_env_tokens": self.total_env_tokens,
            "avg_response_len": round(self.avg_response_len, 1),
            "rollout_time": round(self.rollout_time, 3),
            "llm_time": round(self.llm_time, 3),
            "env_time": round(self.env_time, 3),
            "logprob_time": round(self.logprob_time, 3),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "avg_reward": round(self.avg_reward, 4),
            "rewards": [round(r, 4) for r in self.rewards],
        }


class PerfTracker:
    def __init__(self, output_dir=None, enabled=True):
        self.enabled = enabled
        self.rollout_stats: list[RolloutStats] = []
        self.step_times: list[float] = []
        self.train_start_time = 0.0
        self._current_rollout: RolloutStats | None = None
        self._rollout_start = 0.0
        self._logprob_start = 0.0
        self.output_dir = output_dir
        if output_dir and enabled:
            os.makedirs(output_dir, exist_ok=True)

    def start_training(self):
        self.train_start_time = time.time()

    def start_rollout(self, step):
        self._rollout_start = time.time()
        self._current_rollout = RolloutStats(step=step)

    def end_rollout(self, results):
        if not self._current_rollout:
            return
        rs = self._current_rollout
        rs.rollout_time = time.time() - self._rollout_start
        rs.num_trajectories = len(results)

        for r in results:
            if not isinstance(r, dict):
                continue
            prompt_len = len(r.get("prompt_tokens", []))
            resp_len = len(r.get("response_tokens", []))
            masks = r.get("response_masks", [])
            if hasattr(masks, "tolist"):
                masks = masks.tolist()
            model_tokens = sum(1 for m in masks if m == 1)
            env_tokens = sum(1 for m in masks if m == 0)

            rs.total_prompt_tokens += prompt_len
            rs.total_response_tokens += resp_len
            rs.total_model_tokens += model_tokens
            rs.total_env_tokens += env_tokens
            rs.rewards.append(r.get("trajectory_reward", 0.0))

            metrics = r.get("metrics", {})
            # Use max instead of sum for parallel trajectories
            rs.llm_time = max(rs.llm_time, metrics.get("llm_time", 0))
            rs.env_time = max(rs.env_time, metrics.get("env_time", 0))

        self.rollout_stats.append(rs)
        self._current_rollout = None
        return rs

    def start_logprob_compute(self):
        self._logprob_start = time.time()

    def end_logprob_compute(self):
        if self.rollout_stats:
            self.rollout_stats[-1].logprob_time = time.time() - self._logprob_start

    def record_step_time(self, step_time):
        self.step_times.append(step_time)

    def get_summary(self):
        total_time = time.time() - self.train_start_time if self.train_start_time else 0
        num_steps = len(self.rollout_stats)
        total_trajs = sum(rs.num_trajectories for rs in self.rollout_stats)
        total_rollout_time = sum(rs.rollout_time for rs in self.rollout_stats)
        total_llm_time = sum(rs.llm_time for rs in self.rollout_stats)
        total_logprob_time = sum(rs.logprob_time for rs in self.rollout_stats)
        total_model_tokens = sum(rs.total_model_tokens for rs in self.rollout_stats)
        total_response_tokens = sum(rs.total_response_tokens for rs in self.rollout_stats)
        all_rewards = [r for rs in self.rollout_stats for r in rs.rewards]

        # env_time: cap per-rollout to (rollout_time - llm_time) to avoid
        # overcounting from parallel async trajectories
        total_env_time = 0.0
        for rs in self.rollout_stats:
            max_env = max(rs.rollout_time - rs.llm_time - rs.logprob_time, 0.0)
            total_env_time += min(rs.env_time, max_env)

        train_time = total_time - total_rollout_time if total_time > total_rollout_time else 0

        avg_step_time = total_time / num_steps if num_steps else 0

        return {
            "total_wall_time": round(total_time, 2),
            "total_steps": num_steps,
            "total_trajectories": total_trajs,
            "avg_step_time": round(avg_step_time, 2),
            "time_breakdown": {
                "rollout_total": round(total_rollout_time, 2),
                "rollout_pct": round(total_rollout_time / total_time * 100, 1) if total_time else 0,
                "llm_total": round(total_llm_time, 2),
                "llm_pct": round(total_llm_time / total_time * 100, 1) if total_time else 0,
                "env_total": round(total_env_time, 2),
                "env_pct": round(total_env_time / total_time * 100, 1) if total_time else 0,
                "logprob_total": round(total_logprob_time, 2),
                "logprob_pct": round(total_logprob_time / total_time * 100, 1) if total_time else 0,
                "training_total": round(train_time, 2),
                "training_pct": round(train_time / total_time * 100, 1) if total_time else 0,
            },
            "token_stats": {
                "total_model_tokens": total_model_tokens,
                "total_response_tokens": total_response_tokens,
                "avg_tokens_per_second": round(total_model_tokens / total_llm_time, 1) if total_llm_time else 0,
                "avg_response_length": round(total_response_tokens / total_trajs, 1) if total_trajs else 0,
            },
            "reward_stats": {
                "avg_reward": round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0,
                "min_reward": round(min(all_rewards), 4) if all_rewards else 0,
                "max_reward": round(max(all_rewards), 4) if all_rewards else 0,
            },
        }

    def print_summary(self):
        s = self.get_summary()
        print()
        print("=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"  Wall time:          {self._fmt(s['total_wall_time'])}")
        print(f"  Steps:              {s['total_steps']}")
        print(f"  Trajectories:       {s['total_trajectories']}")
        print(f"  Avg step time:      {self._fmt(s['avg_step_time'])}")
        print()
        tb = s["time_breakdown"]
        print("  Time Breakdown:")
        print(f"    Rollout:          {self._fmt(tb['rollout_total'])} ({tb['rollout_pct']}%)")
        print(f"      LLM inference:  {self._fmt(tb['llm_total'])} ({tb['llm_pct']}%)")
        print(f"      Env execution:  {self._fmt(tb['env_total'])} ({tb['env_pct']}%)")
        print(f"      Logprob comp:   {self._fmt(tb['logprob_total'])} ({tb['logprob_pct']}%)")
        print(f"    Training (GRPO):  {self._fmt(tb['training_total'])} ({tb['training_pct']}%)")
        print()
        ts = s["token_stats"]
        print("  Token Stats:")
        print(f"    Model tokens:     {ts['total_model_tokens']}")
        print(f"    Avg tok/sec:      {ts['avg_tokens_per_second']}")
        print(f"    Avg resp length:  {ts['avg_response_length']}")
        print()
        rs = s["reward_stats"]
        print("  Reward Stats:")
        print(f"    Avg:              {rs['avg_reward']}")
        print(f"    Min:              {rs['min_reward']}")
        print(f"    Max:              {rs['max_reward']}")
        print("=" * 60)

    def save_to_file(self):
        if not self.output_dir or not self.enabled:
            return
        summary = self.get_summary()
        summary["per_step_rollouts"] = [rs.to_dict() for rs in self.rollout_stats]
        filepath = os.path.join(self.output_dir, "perf_stats.json")
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Performance stats saved to {filepath}")

    def _fmt(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"
