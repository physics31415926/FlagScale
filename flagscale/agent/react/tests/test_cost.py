"""Tests for cost tracker."""

from flagscale.agent.react.cost import CostTracker


class TestCostTracker:
    def test_zero_initial(self):
        ct = CostTracker("claude-sonnet-4-20250514")
        assert ct.total_input_tokens == 0
        assert ct.total_output_tokens == 0
        assert ct.estimate_cost() == 0.0

    def test_add_and_estimate(self):
        ct = CostTracker("claude-sonnet-4-20250514")
        ct.add(1_000_000, 100_000)
        cost = ct.estimate_cost()
        assert cost == 1_000_000 * 3.0 / 1_000_000 + 100_000 * 15.0 / 1_000_000
        assert cost == 4.5

    def test_budget_not_exceeded(self):
        ct = CostTracker("claude-sonnet-4-20250514", max_cost=10.0)
        ct.add(1000, 500)
        assert not ct.budget_exceeded()

    def test_budget_exceeded(self):
        ct = CostTracker("claude-sonnet-4-20250514", max_cost=0.01)
        ct.add(100_000, 50_000)
        assert ct.budget_exceeded()

    def test_budget_warning(self):
        ct = CostTracker("claude-sonnet-4-20250514", max_cost=1.0)
        ct.add(200_000, 20_000)
        assert ct.budget_warning()

    def test_no_budget(self):
        ct = CostTracker("claude-sonnet-4-20250514", max_cost=0.0)
        ct.add(1_000_000, 1_000_000)
        assert not ct.budget_exceeded()
        assert not ct.budget_warning()

    def test_unknown_model(self):
        ct = CostTracker("unknown-model-xyz")
        ct.add(1000, 500)
        assert ct.estimate_cost() == 0.0
        assert ct.format_cost() == ""

    def test_format_cost_with_budget(self):
        ct = CostTracker("claude-sonnet-4-20250514", max_cost=5.0)
        ct.add(10000, 2000)
        s = ct.format_cost()
        assert "$" in s
        assert "/ $5.00" in s

    def test_format_cost_no_budget(self):
        ct = CostTracker("gpt-4o")
        ct.add(10000, 2000)
        s = ct.format_cost()
        assert "$" in s
        assert "/" not in s

    def test_partial_model_match(self):
        ct = CostTracker("claude-sonnet-4-20250514-latest")
        ct.add(1_000_000, 0)
        assert ct.estimate_cost() > 0

    def test_custom_pricing(self):
        custom = {"my-custom-model": {"input": 5.0, "output": 20.0}}
        ct = CostTracker("my-custom-model", custom_pricing=custom)
        ct.add(1_000_000, 100_000)
        expected = 1_000_000 * 5.0 / 1_000_000 + 100_000 * 20.0 / 1_000_000
        assert ct.estimate_cost() == expected

    def test_custom_pricing_overrides_builtin(self):
        custom = {"claude-sonnet-4": {"input": 10.0, "output": 50.0}}
        ct = CostTracker("claude-sonnet-4-20250514", custom_pricing=custom)
        ct.add(1_000_000, 0)
        assert ct.estimate_cost() == 10.0

    def test_custom_pricing_empty(self):
        ct = CostTracker("claude-sonnet-4-20250514", custom_pricing={})
        ct.add(1_000_000, 0)
        assert ct.estimate_cost() == 3.0
