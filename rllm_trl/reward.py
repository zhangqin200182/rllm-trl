import re


def math_reward_fn(prompts, completions, **kwargs):
    """TRL-compatible reward function for math problems.

    Extracts the final number from the completion and compares to the ground truth
    answer stored in the dataset's 'answer' field.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        answer = None
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    break
            else:
                content = ""
        else:
            content = str(prompt)

        gt_answer = kwargs.get("answer", [None])[0] if "answer" in kwargs else None

        if isinstance(completion, list):
            completion_text = ""
            for msg in completion:
                if isinstance(msg, dict):
                    completion_text += msg.get("content", "")
        else:
            completion_text = str(completion)

        numbers = re.findall(r'-?\d+\.?\d*', completion_text)
        if numbers and gt_answer is not None:
            try:
                predicted = float(numbers[-1])
                expected = float(gt_answer)
                rewards.append(1.0 if abs(predicted - expected) < 1e-6 else 0.0)
            except (ValueError, TypeError):
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards
