import asyncio
import concurrent.futures
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm_trl.base import Action, BaseAgent, BaseEnv, Trajectory
from rllm_trl.parsers import (
    ChatTemplateParser,
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)

logger = logging.getLogger(__name__)


def compute_trajectory_reward(trajectory: Trajectory) -> Trajectory:
    if not trajectory:
        return trajectory
    trajectory.reward = sum(d.reward for d in trajectory.steps)
    return trajectory


def compute_mc_return(trajectory: Trajectory, gamma: float = 0.95) -> Trajectory:
    G = 0.0
    for step in reversed(trajectory.steps):
        G = step.reward + gamma * G
        step.mc_return = G
    return trajectory


class HFAgentExecutionEngine:
    def __init__(
        self,
        model,
        tokenizer,
        chat_parser=None,
        n_parallel_agents=1,
        gamma=0.2,
        retry_limit=3,
        max_steps=5,
        max_response_length=512,
        max_prompt_length=512,
        agent_class=None,
        env_class=None,
        agent_args=None,
        env_args=None,
        max_workers=8,
        sampling_params=None,
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if env_args is None:
            env_args = {}
        if sampling_params is None:
            sampling_params = {}

        self.model = model
        self.tokenizer = tokenizer
        self.n_parallel_agents = n_parallel_agents
        self.gamma = gamma
        self.retry_limit = retry_limit
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args
        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.sampling_params = sampling_params
        self.on_trajectory_done = kwargs.get("on_trajectory_done", None)

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(
                self.tokenizer,
                disable_thinking=kwargs.get("disable_thinking", False),
            )
        else:
            self.chat_parser = chat_parser

    def update_model(self, model):
        self.model = model

    def update_envs_and_agents(self, envs, agents):
        assert len(agents) == len(envs)
        self.envs = envs
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)

    async def get_model_response(self, prompt, **kwargs):
        prompt_text = prompt
        if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
            prompt_text = self.chat_parser.parse(
                prompt, add_generation_prompt=True, is_first_msg=True
            )

        input_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        max_new_tokens = kwargs.get("max_tokens", 256)
        max_new_tokens = min(max_new_tokens, self.max_response_length)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.sampling_params.get("temperature", 0.7),
                top_p=self.sampling_params.get("top_p", 0.9),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response

    async def run_agent_trajectory_async(self, idx, seed=0, mode="Token", **kwargs):
        agent = self.agents[idx]
        env = self.envs[idx]

        prompt_tokens = []
        response_tokens = []
        response_masks = []
        response_token_len = 0
        llm_time = 0.0
        env_time = 0.0
        total_time = 0.0
        reward_time = None
        reward = 0.0
        termination_reason = None

        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        agent.reset()
        agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)

        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(
            messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=True,
            contains_generation_msg=True,
        )

        for step_idx in range(self.max_steps):
            prompt_messages = agent.chat_completions.copy()
            max_tokens = self.max_response_length - response_token_len
            if max_tokens <= 0:
                termination_reason = "TRUNCATION"
                break

            start_time = time.time()
            response = await self.get_model_response(prompt_messages, max_tokens=max_tokens)
            delta = time.time() - start_time
            llm_time += delta
            total_time += delta

            action: Action = agent.update_from_model(response)
            action = action.action

            start_time = time.time()
            try:
                next_obs, reward, done, info = await loop.run_in_executor(
                    self.executor, env.step, action
                )
            except Exception:
                traceback.print_exc()
                reward = 0
                done = True
                termination_reason = "ENV_ERROR"
                break

            delta = time.time() - start_time
            env_time += delta
            total_time += delta
            info["max_steps"] = self.max_steps

            agent.update_from_env(observation=next_obs, reward=reward, done=done, info=info)

            cur_step = agent.get_current_state()
            if cur_step:
                cur_step.reward = reward
                cur_step.done = done

            chat_msgs = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_msgs)

            asst_tokens, asst_masks = [], []
            env_tokens, env_masks = [], []
            if assistant_message:
                asst_tokens, asst_masks = convert_messages_to_tokens_and_masks(
                    [assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser,
                    contains_first_msg=False, contains_generation_msg=False,
                )
            if env_messages:
                env_tokens, env_masks = convert_messages_to_tokens_and_masks(
                    env_messages, tokenizer=self.tokenizer, parser=self.chat_parser,
                    contains_first_msg=False, contains_generation_msg=True,
                )

            response_token_len += len(asst_tokens) + len(env_tokens)

            if response_token_len >= self.max_response_length:
                trunc = self.max_response_length - response_token_len
                if trunc < 0:
                    combined_t = (asst_tokens + env_tokens)[:trunc]
                    combined_m = (asst_masks + env_masks)[:trunc]
                else:
                    combined_t = asst_tokens + env_tokens
                    combined_m = asst_masks + env_masks
                response_tokens.extend(combined_t)
                response_masks.extend(combined_m)
                termination_reason = "TRUNCATION"
                break

            response_tokens.extend(asst_tokens)
            response_masks.extend(asst_masks)

            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_tokens)
            response_masks.extend(env_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        if hasattr(env, "compute_final_reward"):
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step = agent.get_current_state()
            if cur_step:
                cur_step.reward = reward

        await loop.run_in_executor(self.executor, env.close)

        if termination_reason:
            logger.info(f"Trajectory {idx}: {termination_reason}, reward={reward}")

        trajectory: Trajectory = agent.trajectory
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Token":
            return {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "metrics": {
                    "steps": len(trajectory.steps),
                    "reward_time": reward_time,
                    "env_time": env_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                },
            }
        return trajectory

    async def run_trajectories(self, mode="Token", **kwargs):
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_agents)
        tasks = [
            self.run_agent_trajectory_async(idx=i, mode=mode, **kwargs)
            for i in range(len(self.envs))
        ]
        results = []
        total = len(self.envs)
        completed = 0
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1
                if self.on_trajectory_done:
                    reward = result.get("trajectory_reward", 0.0) if isinstance(result, dict) else 0.0
                    self.on_trajectory_done(completed, total, reward)
            except Exception:
                traceback.print_exc()
        self.executor.shutdown(wait=False, cancel_futures=True)
        results.sort(key=lambda x: x["idx"] if isinstance(x, dict) else 0)
        return results
