"""
Shared constants and helpers for ScienceWorld EvolEval runners.

Key differences from AlfWorld/BabyAI:
  - env = ScienceWorldEnv(task_name, variation, step_limit=100)
  - env.step(action_str) takes free-form text strings (like Jericho/AlfWorld)
  - task_name and variation parsed directly from episode ID via regex
  - No game_file cache or seed cache needed
  - Scoring: info["progress"] (float 0-1, subgoal-based) and info["won"]
  - info["action_templates"] provides grammar patterns for agent prompts
  - STEP_LIMIT = 100
"""

import sys
import json
import re
from pathlib import Path

SCIWORLD_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(SCIWORLD_PATH))
from src.scienceworld_env import ScienceWorldEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text  # noqa: F401

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATASET_ROOT = SCIWORLD_PATH.parent / "final_dataset" / "scienceworld"
RESULTS_ROOT = Path(__file__).parent / "results"

MAX_CONTEXT_TOKENS = 32000
STEP_LIMIT = 100

GAME_LIST = [
    "boil",
    "chemistry-mix",
    "find-living-thing",
    "grow-plant",
    "inclined-plane-friction-named-surfaces",
    "measure-melting-point-known-substance",
    "melt",
    "power-component",
    "test-conductivity-of-unknown-substances",
    "use-thermometer",
]


# ---------------------------------------------------------------------------
# Episode ID parsing (task_name and variation encoded in ID)
# ---------------------------------------------------------------------------

def parse_task_and_variation(ep_id: str) -> tuple[str, int]:
    """
    Parse task_name and variation from a ScienceWorld episode ID.

    Format: scienceworld_{task_name}_var{NNN}_ep{NN}_{model}
    Example: scienceworld_inclined-plane-friction-named-surfaces_var003_ep01_Qwen_Qwen3-32B
    """
    m = re.match(r"scienceworld_(.+)_var(\d+)_ep\d+", ep_id)
    if m is None:
        raise ValueError(f"Cannot parse task/variation from episode ID: {ep_id!r}")
    task_name = m.group(1)
    variation = int(m.group(2))
    return task_name, variation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snapshots(game: str = None) -> dict[str, dict]:
    """Load episodes indexed by episode ID. If game is None, loads all tasks."""
    games_to_load = GAME_LIST if game is None else [game]
    result: dict[str, dict] = {}
    for g in games_to_load:
        path = DATASET_ROOT / g / "snapshots.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        result.update({ep["id"]: ep for ep in data["episodes"]})
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_trajectory(trajectory: list[dict], header: str = None) -> str:
    """Format a trajectory for inclusion in LLM prompts."""
    lines = []
    if header:
        lines.append(f"=== {header} ===")
    for step in trajectory:
        lines.append(f"[Step {step['step']}] Observation: {step['obs']}")
        if step.get("action"):
            lines.append(f"           Action taken: {step['action']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Game state restore
# ---------------------------------------------------------------------------

def restore_game_state(env: ScienceWorldEnv,
                       trajectory: list[dict],
                       decision_point: int):
    """
    Reset env and replay trajectory[1:dp+1] to arrive at the state at decision_point.

    ScienceWorld trajectory convention (same as AlfWorld/BabyAI):
      step 0: initial obs from reset(), action=None
      step i (i>0): obs RECEIVED after taking step[i]["action"]

    env must already have been constructed with the correct task_name and variation.
    """
    ob, info = env.reset()
    for step_data in trajectory[1:decision_point + 1]:
        action = step_data.get("action")
        if not action:
            continue
        ob, done, info = env.step(action)
        if done:
            break
    return ob, info
