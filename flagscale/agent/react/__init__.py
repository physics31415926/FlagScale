"""FlagScale ReAct Agent."""

from flagscale.agent.react.agent import ReactAgent
from flagscale.agent.react.config import AgentConfig


def run_agent(provider: str = "anthropic", model: str = None):
    """Entry point: create and run the agent."""
    config = AgentConfig.auto_load(provider=provider, model=model)
    agent = ReactAgent(config)
    agent.run()
