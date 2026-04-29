import asyncio
import logging
import time

import torch

from rllm_trl.hf_engine import HFAgentExecutionEngine
from rllm_trl.math_env import MathCalcEnv

logger = logging.getLogger(__name__)


def make_rllm_rollout_func(
    agent_class,
    agent_args=None,
    env_class=None,
    env_args=None,
    max_steps=3,
    max_response_length=256,
    max_prompt_length=512,
    sampling_params=None,
    training_logger=None,
    trajectory_writer=None,
    perf_tracker=None,
    answer_map=None,
):
    if agent_args is None:
        agent_args = {}
    if env_args is None:
        env_args = {}
    if env_class is None:
        env_class = MathCalcEnv
    if sampling_params is None:
        sampling_params = {"temperature": 0.7, "top_p": 0.9}
    if answer_map is None:
        answer_map = {}

    _step_counter = [0]

    def rollout_func(prompts, trainer):
        _step_counter[0] += 1
        step = _step_counter[0]
        model = trainer.model
        tokenizer = trainer.processing_class
        device = trainer.accelerator.device

        num_gens = getattr(trainer.args, "num_generations", len(prompts))
        if training_logger:
            training_logger.log_rollout_start(step, len(prompts), num_gens)
        if perf_tracker:
            perf_tracker.start_rollout(step)

        model.eval()

        engine = HFAgentExecutionEngine(
            model=model,
            tokenizer=tokenizer,
            n_parallel_agents=len(prompts),
            max_steps=max_steps,
            max_response_length=max_response_length,
            max_prompt_length=max_prompt_length,
            agent_class=agent_class,
            agent_args=agent_args,
            env_class=env_class,
            env_args=env_args,
            sampling_params=sampling_params,
            on_trajectory_done=lambda idx, total, reward: (
                training_logger.log_trajectory_done(step, idx, total, reward)
                if training_logger else None
            ),
        )

        envs = []
        agents = []
        for prompt in prompts:
            task = _extract_task_from_prompt(prompt)
            if task["question"] in answer_map:
                task["answer"] = answer_map[task["question"]]
            task.update(env_args)
            env = env_class.from_dict(task)
            agent = agent_class(**agent_args)
            envs.append(env)
            agents.append(agent)

        engine.update_envs_and_agents(envs, agents)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(
                        lambda: asyncio.run(engine.run_trajectories(mode="Token"))
                    ).result()
            else:
                results = loop.run_until_complete(engine.run_trajectories(mode="Token"))
        except RuntimeError:
            results = asyncio.run(engine.run_trajectories(mode="Token"))

        model.train()

        # Perf tracking: end rollout phase
        rollout_stats = None
        if perf_tracker:
            rollout_stats = perf_tracker.end_rollout(results)

        # Write trajectories to file
        if trajectory_writer:
            trajectory_writer.write_rollout(step, agents, envs, results, tokenizer)

        prompt_ids_list = []
        completion_ids_list = []
        env_mask_list = []

        for result in results:
            prompt_ids_list.append(result["prompt_tokens"].tolist())
            completion_ids_list.append(result["response_tokens"].tolist())
            env_mask_list.append(result["response_masks"].tolist())

        rewards = [r.get("trajectory_reward", 0.0) if isinstance(r, dict) else 0.0 for r in results]
        if training_logger:
            training_logger.log_rollout_done(step, len(results), rewards, rollout_stats=rollout_stats)

        # Compute logprobs with timing
        if training_logger:
            training_logger.log_logprob_start(step)
        if perf_tracker:
            perf_tracker.start_logprob_compute()

        logprobs_list = _compute_logprobs(
            model, tokenizer, prompt_ids_list, completion_ids_list, device
        )

        if perf_tracker:
            perf_tracker.end_logprob_compute()

        if training_logger:
            training_logger.log_training_update_start(step)

        return {
            "prompt_ids": prompt_ids_list,
            "completion_ids": completion_ids_list,
            "logprobs": logprobs_list,
            "env_mask": env_mask_list,
        }

    return rollout_func


def _extract_task_from_prompt(prompt):
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return {"question": msg["content"]}
        return {"question": str(prompt)}
    return {"question": str(prompt)}


def _compute_logprobs(model, tokenizer, prompt_ids_list, completion_ids_list, device):
    logprobs_list = []
    for prompt_ids, completion_ids in zip(prompt_ids_list, completion_ids_list):
        if not completion_ids:
            logprobs_list.append([])
            continue
        input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        prompt_len = len(prompt_ids)
        completion_logits = logits[0, prompt_len - 1:-1, :]
        completion_tokens = torch.tensor(completion_ids, device=device)
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        token_logprobs = log_probs.gather(1, completion_tokens.unsqueeze(1)).squeeze(1)
        logprobs_list.append(token_logprobs.cpu().tolist())
    return logprobs_list
