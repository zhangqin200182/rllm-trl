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


def generate_math_problems(n=100, seed=42, difficulty="mixed"):
    rng = random.Random(seed)
    problems = []

    def _simple(rng):
        ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        sym, fn = rng.choice(ops)
        return f"What is {a} {sym} {b}?", str(fn(a, b))

    def _multi_step(rng):
        templates = [
            lambda: _multi_step_chain(rng),
            lambda: _word_problem(rng),
            lambda: _percentage_problem(rng),
            lambda: _comparison_problem(rng),
        ]
        return rng.choice(templates)()

    def _multi_step_chain(rng):
        a, b, c = rng.randint(2, 50), rng.randint(2, 50), rng.randint(2, 20)
        op1, op2 = rng.choice([("+", "-"), ("*", "+"), ("+", "*"), ("-", "+"), ("*", "-")])
        expr = f"({a} {op1} {b}) {op2} {c}"
        answer = eval(expr)  # noqa: S307
        patterns = [
            f"First compute {a} {op1} {b}, then {op2} {c}. What is the result?",
            f"What is ({a} {op1} {b}) {op2} {c}?",
            f"Calculate: start with {a}, {_op_word(op1)} {b}, then {_op_word(op2)} {c}.",
        ]
        return rng.choice(patterns), str(answer)

    def _word_problem(rng):
        items = [("apples", "oranges"), ("books", "pens"), ("shirts", "pants"), ("tickets", "drinks")]
        item1, item2 = rng.choice(items)
        p1, p2 = rng.randint(2, 15), rng.randint(2, 15)
        q1, q2 = rng.randint(1, 10), rng.randint(1, 10)
        total = p1 * q1 + p2 * q2
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        name = rng.choice(names)
        question = (
            f"{name} buys {q1} {item1} at ${p1} each and {q2} {item2} at ${p2} each. "
            f"How much does {name} spend in total?"
        )
        return question, str(total)

    def _percentage_problem(rng):
        base = rng.choice([50, 80, 100, 120, 150, 200, 250, 300, 400, 500])
        pct = rng.choice([10, 15, 20, 25, 30, 40, 50, 75])
        result = base * pct / 100
        patterns = [
            f"What is {pct}% of {base}?",
            f"A product costs ${base}. If there is a {pct}% discount, how much do you save?",
            f"Calculate {pct} percent of {base}.",
        ]
        answer = int(result) if result == int(result) else result
        return rng.choice(patterns), str(answer)

    def _comparison_problem(rng):
        a, b = rng.randint(5, 50), rng.randint(5, 50)
        c, d = rng.randint(1, 30), rng.randint(1, 30)
        val1, val2 = a * b, c * d
        question = (
            f"Store A sells {a} items at ${b} each. Store B sells {c} items at ${d} each. "
            f"How much more does the store with higher revenue earn?"
        )
        return question, str(abs(val1 - val2))

    def _op_word(op):
        return {"+" : "add", "-": "subtract", "*": "multiply by"}.get(op, op)

    for _ in range(n):
        if difficulty == "simple":
            q, a = _simple(rng)
        elif difficulty == "hard":
            q, a = _multi_step(rng)
        else:
            q, a = _multi_step(rng) if rng.random() > 0.8 else _simple(rng)
        problems.append({"question": q, "answer": str(a)})
    return problems
