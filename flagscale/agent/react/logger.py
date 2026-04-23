"""Logging configuration for the agent."""

import logging
import os
import sys


def setup_logging():
    """Configure logging to stderr so it doesn't interfere with user interaction."""
    level_name = os.environ.get("FLAGSCALE_AGENT_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s", datefmt="%H:%M:%S")
    )

    root = logging.getLogger("flagscale.agent")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
