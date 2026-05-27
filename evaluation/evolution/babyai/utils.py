"""
Shared constants and helpers for BabyAI EvolEval runners.

Key differences from Jericho/AlfWorld:
  - Actions are strings (from ACTIONS list), but env.step() takes integer index
  - instance_seed not stored in final_dataset — looked up from BabyAI/output/ at runtime
  - obs_to_reward patterns loaded from data/babyai/test.jsonl for progress tracking
  - env = BabyAIEnv(level_id, obs_to_reward=..., seed=instance_seed)
  - env.reset() uses stored seed; env.step(action_idx) takes int 0-6
  - Scoring: info["progress"] (float 0-1) and info["won"]
  - No inventory field
"""

import sys
import json
import glob
import re
from pathlib import Path

BABYAI_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(BABYAI_PATH))
from src.babyai_env import BabyAIEnv, ACTIONS, GAME_LEVELS
from src.openai_helpers import chat_completion_with_retries, truncate_text  # noqa: F401

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATASET_ROOT = BABYAI_PATH.parent / "final_dataset" / "babyai"
SNAPSHOTS_DIR = BABYAI_PATH / "output"
LABEL_PATH = BABYAI_PATH / "data" / "babyai" / "test.jsonl"
RESULTS_ROOT = Path(__file__).parent / "results"

MAX_CONTEXT_TOKENS = 32000
STEP_LIMIT = 128

GAME_LIST = [f"level-{i}" for i in range(1, 21)]

# Valid action strings shown to the LLM
ACTIONS_PROMPT = " | ".join(ACTIONS)

# ---------------------------------------------------------------------------
# Labels (obs_to_reward patterns for progress tracking)
# ---------------------------------------------------------------------------

_labels_cache: dict[int, list] | None = None

def load_labels() -> dict[int, list]:
    """Load obs_to_reward patterns for all levels from AgentBoard label file."""
    global _labels_cache
    if _labels_cache is not None:
        return _labels_cache
    _labels_cache = {}
    if LABEL_PATH.exists():
        with open(LABEL_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                _labels_cache[e["level"]] = e.get("obs_to_reward", [])
    return _labels_cache


# ---------------------------------------------------------------------------
# Seed cache (instance_seed not stored in final_dataset)
# ---------------------------------------------------------------------------

_seed_cache: dict[str, dict] | None = None

def _build_seed_cache() -> dict[str, dict]:
    """Scan original BabyAI/output snapshots and build ep_id → {seed, level_id}."""
    cache: dict[str, dict] = {}
    for path in sorted(SNAPSHOTS_DIR.rglob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        orig_id = snap.get("id", "")
        model_clean = snap.get("model", "").replace("/", "_")
        seed = snap.get("instance_seed")
        level_id = snap.get("level_id")
        if not orig_id or seed is None:
            continue
        uid = f"{orig_id}_{model_clean}"
        cache[uid] = {"seed": seed, "level_id": level_id}
    return cache


def get_seed_and_level(ep_id: str) -> tuple[int, int]:
    """Return (instance_seed, level_id) for a dataset episode ID."""
    global _seed_cache
    if _seed_cache is None:
        _seed_cache = _build_seed_cache()
    entry = _seed_cache.get(ep_id)
    if entry is None:
        raise ValueError(f"instance_seed not found for episode: {ep_id!r}")
    return entry["seed"], entry["level_id"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snapshots(game: str = None) -> dict[str, dict]:
    """Load episodes indexed by episode ID. If game is None, loads all levels."""
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
# Action helpers
# ---------------------------------------------------------------------------

def action_str_to_idx(action: str) -> int:
    """Convert action string to BabyAI env integer index. Falls back to 'move forward'."""
    action = action.strip().lower()
    for i, a in enumerate(ACTIONS):
        if a.lower() == action:
            return i
    # partial match fallback
    for i, a in enumerate(ACTIONS):
        if action in a.lower() or a.lower() in action:
            return i
    return ACTIONS.index("move forward")  # safe default


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

def restore_game_state(env: BabyAIEnv,
                       trajectory: list[dict],
                       decision_point: int):
    """
    Reset env and replay trajectory[1:dp+1] to arrive at the state at decision_point.

    BabyAI trajectory convention (same as AlfWorld):
      step 0: initial obs from reset(), action=None
      step i (i>0): obs RECEIVED after taking step[i]["action"]

    env must already have been constructed with the correct seed.
    """
    ob, info = env.reset()
    for step_data in trajectory[1:decision_point + 1]:
        action = step_data.get("action")
        if not action:
            continue
        action_idx = action_str_to_idx(action)
        ob, done, info = env.step(action_idx)
        if done:
            break
    return ob, info
