import random
import re

from rllm_trl.base import BaseEnv


class MathCalcEnv(BaseEnv):
    """Simple math calculation environment with a calculator tool."""

    def __init__(self, task=None, max_steps=3):
        self.task = task or {}
        self.max_steps = max_steps
        self.step_count = 0
        self.question = self.task.get("question", "")
        self.answer = self.task.get("answer", "")

    def reset(self):
        self.step_count = 0
        observation = {"question": self.question}
        return observation, {}

    def step(self, action):
        self.step_count += 1
        done = False

        if isinstance(action, str):
            done = True
            reward = self._check_answer(action)
            return {}, reward, done, {}

        if isinstance(action, list):
            for tool_call in action:
                func = tool_call.get("function", {})
                if func.get("name") == "finish":
                    done = True
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        import json
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"response": args}
                    response = args.get("response", "")
                    reward = self._check_answer(response)
                    return {}, reward, done, {}

            tool_outputs = {}
            for tool_call in action:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                if name == "calculate":
                    expr = args.get("expression", "")
                    result = self._safe_eval(expr)
                    tool_outputs[tool_call.get("id", "0")] = str(result)
                else:
                    tool_outputs[tool_call.get("id", "0")] = f"Unknown tool: {name}"

            if self.step_count >= self.max_steps:
                done = True

            return {"tool_outputs": tool_outputs}, 0.0, done, {}

        done = True
        return {}, 0.0, done, {}

    def _check_answer(self, response):
        numbers = re.findall(r'-?\d+\.?\d*', str(response))
        if not numbers:
            return 0.0
        predicted = float(numbers[-1])
        try:
            expected = float(self.answer)
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if abs(predicted - expected) < 1e-6 else 0.0

    def _safe_eval(self, expr):
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in str(expr)):
                return "Error: invalid expression"
            return eval(str(expr))  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    def close(self):
        pass

    @staticmethod
    def from_dict(info):
        return MathCalcEnv(task=info, max_steps=info.get("max_steps", 3))

    @staticmethod
    def is_multithread_safe():
        return True


def generate_math_problems(n=100, seed=42):
    rng = random.Random(seed)
    problems = []
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    for _ in range(n):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        op_sym, op_fn = rng.choice(ops)
        answer = op_fn(a, b)
        question = f"What is {a} {op_sym} {b}?"
        problems.append({"question": question, "answer": str(answer)})
    return problems
