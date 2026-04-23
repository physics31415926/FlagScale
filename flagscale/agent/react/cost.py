"""Token cost estimation and budget control."""

PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-haiku-4": {"input": 0.80, "output": 4.0},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
}


class CostTracker:
    """Track token usage and estimate costs in USD."""

    def __init__(self, model: str, max_cost: float = 0.0):
        self._model = model
        self._max_cost = max_cost
        self._total_input = 0
        self._total_output = 0

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    def _get_pricing(self):
        for key, pricing in PRICING.items():
            if key in self._model:
                return pricing
        return None

    def add(self, input_tokens: int, output_tokens: int):
        self._total_input += input_tokens
        self._total_output += output_tokens

    def estimate_cost(self) -> float:
        pricing = self._get_pricing()
        if not pricing:
            return 0.0
        return (
            self._total_input * pricing["input"] / 1_000_000
            + self._total_output * pricing["output"] / 1_000_000
        )

    def budget_exceeded(self) -> bool:
        if self._max_cost <= 0:
            return False
        return self.estimate_cost() >= self._max_cost

    def budget_warning(self) -> bool:
        if self._max_cost <= 0:
            return False
        return self.estimate_cost() >= self._max_cost * 0.8

    def format_cost(self) -> str:
        cost = self.estimate_cost()
        if cost == 0.0 and not self._get_pricing():
            return ""
        s = f"${cost:.4f}"
        if self._max_cost > 0:
            s += f" / ${self._max_cost:.2f}"
        return s
