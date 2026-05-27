"""
Shared constants and helpers for BundledWebShopping EvolEval runners.

Key differences from step-based environments (AlfWorld, ScienceWorld):
  - env.step() processes ONE complete subtask per call, not one game action
  - Trajectory uses paired steps: even = pre-action obs (action=None),
    odd = post-action obs (action = selection letter + reasoning)
  - decision_point is always even; decision_point//2 = failing subtask index
  - restore_game_state uses direct state setting (no action replay needed)
  - STEP_LIMIT = max subtasks per task (safety bound; env returns done=True naturally)
"""

import sys
import re
import json
from pathlib import Path

BUNDLED_WEB_SHOPPING_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BUNDLED_WEB_SHOPPING_PATH))
from src.bundled_shopping_env import BundledShoppingEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text  # noqa: F401

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).parent.parent.parent / "final_dataset" / "bundled_web_shopping"
RESULTS_ROOT = Path(__file__).parent / "results"

GAME_LIST = ["baking", "beauty", "electronics", "grocery", "home"]
STEP_LIMIT = 20        # safety bound; env exits via done=True
MAX_CONTEXT_TOKENS = 32000

# ---------------------------------------------------------------------------
# Dataset cache
# ---------------------------------------------------------------------------

_dataset_cache = None


def load_dataset_split():
    global _dataset_cache
    if _dataset_cache is None:
        from datasets import load_dataset
        _dataset_cache = load_dataset(
            "ZexueHe/memoryarena", "bundled_shopping", trust_remote_code=True
        )["test"]
    return _dataset_cache


# ---------------------------------------------------------------------------
# Episode ID helpers
# ---------------------------------------------------------------------------

def parse_task_id(ep_id: str) -> int:
    """Extract integer task index from episode ID like bundled_shopping_task042_ep00_..."""
    m = re.match(r"bundled_shopping_task(\d+)_ep\d+", ep_id)
    if m is None:
        raise ValueError(f"Cannot parse task_id from episode ID: {ep_id!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snapshots(game: str = None) -> dict[str, dict]:
    """Load episodes indexed by episode ID. Pass game=None to load all games."""
    games_to_load = [game] if game else GAME_LIST
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
# Env creation
# ---------------------------------------------------------------------------

def make_env(ep_id: str, dataset=None) -> BundledShoppingEnv:
    task_id = parse_task_id(ep_id)
    if dataset is None:
        dataset = load_dataset_split()
    return BundledShoppingEnv(task_id, dataset)


# ---------------------------------------------------------------------------
# Game state restoration
# ---------------------------------------------------------------------------

def restore_game_state(env: BundledShoppingEnv,
                       trajectory: list[dict],
                       decision_point: int):
    """
    Fast-forward env to the failing subtask without replaying actions.

    decision_point is always even. decision_point//2 is the subtask index.
    We set env._current directly to avoid redundant env.step() calls.
    Returns (obs, info) where obs = trajectory[decision_point]["obs"].
    """
    env.reset()
    env._current = decision_point // 2
    env._decisions = []
    ob = trajectory[decision_point]["obs"]
    return ob, {}


# ---------------------------------------------------------------------------
# Context formatting helpers
# ---------------------------------------------------------------------------

def format_prior_subtasks(trajectory: list[dict], decision_point: int) -> str:
    """Format pre-decision-point subtask history for agent context."""
    lines = []
    for i in range(decision_point // 2):
        even_step = i * 2
        odd_step = i * 2 + 1
        if odd_step >= len(trajectory):
            break
        action = trajectory[odd_step].get("action", "")
        correct = trajectory[odd_step].get("correct", None)
        # Extract the selection letter
        m = re.search(r"Selection:\s*\[([A-Za-z])\]|\[([A-Za-z])\]", action)
        sel = f"[{(m.group(1) or m.group(2)).upper()}]" if m else "(unknown)"
        status = " ✓" if correct else " ✗" if correct is not None else ""
        lines.append(f"Subtask {i + 1}: Selected {sel}{status}")
    return "\n".join(lines)


def format_trajectory_for_evo(trajectory: list[dict], final_score: float) -> str:
    """Format a complete episode trajectory for evotest evolution calls."""
    lines = ["EPISODE HISTORY:"]
    for step in trajectory:
        if step.get("action") is None:
            continue  # skip even (pre-action) steps
        subtask_idx = step.get("subtask_idx", step.get("step", 0) // 2)
        obs = step["obs"]
        action = step.get("action", "")
        correct = step.get("correct", None)
        lines.append(f"\n[Subtask {subtask_idx + 1}]")
        lines.append(f"Question: {obs[:300]}")
        lines.append(f"Selected: {action[:150]}")
        if correct is not None:
            lines.append(f"Result: {'CORRECT' if correct else 'INCORRECT'}")
        lines.append("---")
    lines.append(f"\nFinal score: {final_score:.2f}")
    return "\n".join(lines)
