"""
Shared constants and helpers for GroupTravelPlanning EvolEval runners.

Key differences from step-based environments:
  - env.step() processes ONE traveler's plan per call (free-text generation)
  - env.step() makes an LLM judge call — expensive! restore_game_state uses
    direct state setting to avoid re-judging historical subtasks
  - Trajectory uses paired steps: even = pre-action obs (action=None),
    odd = post-action obs (action = generated plan text)
  - decision_point is always even; decision_point//2 = failing traveler index
  - Prior travelers' plans must be shown as context (RELATION/JOIN constraints
    reference other travelers' specific restaurant/accommodation choices)
"""

import sys
import re
import json
from pathlib import Path

GROUP_TRAVEL_PLANNING_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(GROUP_TRAVEL_PLANNING_PATH))
from src.group_travel_env import GroupTravelPlanningEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text  # noqa: F401

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).parent.parent.parent / "final_dataset" / "group_travel_planning"
RESULTS_ROOT = Path(__file__).parent / "results"

GAME_LIST = ["5_travelers", "6_travelers", "7_travelers", "8_travelers"]
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
        _dataset_cache = load_dataset("ZexueHe/memoryarena", "group_travel_planner")["test"]
    return _dataset_cache


# ---------------------------------------------------------------------------
# Episode ID helpers
# ---------------------------------------------------------------------------

def parse_task_id(ep_id: str) -> int:
    """Extract integer task index from episode ID like group_travel_task042_ep00_..."""
    m = re.match(r"group_travel_task(\d+)_ep\d+", ep_id)
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

def make_env(ep_id: str, judge_model: str, dataset=None) -> GroupTravelPlanningEnv:
    task_id = parse_task_id(ep_id)
    if dataset is None:
        dataset = load_dataset_split()
    return GroupTravelPlanningEnv(task_id, dataset, judge_model)


# ---------------------------------------------------------------------------
# Game state restoration
# ---------------------------------------------------------------------------

def restore_game_state(env: GroupTravelPlanningEnv,
                       trajectory: list[dict],
                       decision_point: int):
    """
    Fast-forward env to the failing traveler without calling the LLM judge.

    Each env.step() call makes an LLM judge API call — replaying historical
    steps would be extremely expensive. Instead, set env._current directly.
    Returns (obs, info) where obs = trajectory[decision_point]["obs"]
    (the full env observation for the failing traveler).
    """
    env.reset()
    env._current = decision_point // 2
    env._decisions = []
    ob = trajectory[decision_point]["obs"]
    return ob, {}


# ---------------------------------------------------------------------------
# Context formatting helpers
# ---------------------------------------------------------------------------

def format_prior_plans(trajectory: list[dict], decision_point: int) -> str:
    """
    Format pre-decision-point traveler plans for agent context.
    Shows each prior traveler's constraints and the generated plan,
    which is critical for RELATION/JOIN constraints in later travelers.
    """
    lines = []
    for i in range(decision_point // 2):
        odd_step = i * 2 + 1
        if odd_step >= len(trajectory):
            break
        question = trajectory[odd_step].get("obs", "")  # raw question text
        plan = trajectory[odd_step].get("action", "")
        # Strip thinking tags from plan
        plan_clean = re.sub(r"<think>.*?</think>", "", plan, flags=re.DOTALL).strip()
        prog = float(trajectory[odd_step].get("subtask_progress", 0) or 0)
        lines.append(f"=== Traveler {i + 1} ===")
        lines.append(f"Request: {question[:400]}")
        lines.append(f"Plan: {plan_clean[:600]}")
        lines.append(f"Constraint satisfaction: {prog:.0%}")
        lines.append("")
    return "\n".join(lines)


def format_trajectory_for_evo(trajectory: list[dict], final_score: float) -> str:
    """Format a complete episode trajectory for evotest evolution calls."""
    lines = ["EPISODE HISTORY:"]
    for step in trajectory:
        if step.get("action") is None:
            continue  # skip even (pre-action) steps
        subtask_idx = step.get("subtask_idx", step.get("step", 0) // 2)
        obs = step.get("obs", "")  # raw question text
        action = step.get("action", "")
        action_clean = re.sub(r"<think>.*?</think>", "", action, flags=re.DOTALL).strip()
        prog = float(step.get("subtask_progress", 0) or 0)
        lines.append(f"\n[Traveler {subtask_idx + 1}]")
        lines.append(f"Request: {obs[:300]}")
        lines.append(f"Plan: {action_clean[:400]}")
        lines.append(f"Score: {prog:.0%}")
        lines.append("---")
    lines.append(f"\nFinal score: {final_score:.2f}")
    return "\n".join(lines)
