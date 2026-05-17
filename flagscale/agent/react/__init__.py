"""FlagScale Agent. Single WorkerAgent with composable Interrupt + Checklist + Judge architecture.

Entry points:
- run_agent(provider, model, mode) — high-level CLI launcher
"""

from flagscale.agent.react.agent import WorkerAgent
from flagscale.agent.react.config import AgentConfig
from flagscale.agent.react.scene import ScenePreset, PRESETS
from flagscale.agent.react.profile import WorkerProfile, PROFILES


def run_agent(provider: str = "anthropic", model: str = None, mode: str = None):
    """Entry point: create and run the agent."""
    config = AgentConfig.auto_load(provider=provider, model=model, mode=mode)
    agent = WorkerAgent(config)
    agent.run()
