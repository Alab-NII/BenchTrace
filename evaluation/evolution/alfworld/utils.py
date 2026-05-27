"""
Shared constants and helpers for AlfWorld EvolEval runners.

Key differences from JTTL/EvolEval/utils.py:
  - env.step() returns (obs, done, info) — no reward in return tuple
  - restore_game_state replays trajectory[1:dp+1] (step 0 has action=None)
  - Scoring uses info["progress"] (float 0-1) and info["won"], not integer score
  - game_file not stored in final dataset — looked up from original snapshots at runtime
  - No inventory field in trajectory or info
"""

import sys
import json
import re
from pathlib import Path

ALFWORLD_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ALFWORLD_PATH))
from src.alfworld_env import AlfWorldEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text  # noqa: F401

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).parent.parent.parent / "final_dataset" / "alfworld"
SNAPSHOTS_DIR = ALFWORLD_PATH / "output"
RESULTS_ROOT = Path(__file__).parent / "results"

MAX_CONTEXT_TOKENS = 32000
STEP_LIMIT = 50

TASK_SHORTS = {
    "pick_and_place_simple":           "pick_and_place",
    "look_at_obj_in_light":            "look_at_obj",
    "pick_clean_then_place_in_recep":  "pick_clean",
    "pick_heat_then_place_in_recep":   "pick_heat",
    "pick_cool_then_place_in_recep":   "pick_cool",
    "pick_two_obj_and_place":          "pick_two",
}

TASK_TYPES = list(TASK_SHORTS.keys())
TASK_SHORT_LIST = list(TASK_SHORTS.values())

# ---------------------------------------------------------------------------
# Game-file lookup  (game_file is not stored in final dataset)
# ---------------------------------------------------------------------------

_game_file_cache: dict[str, str] | None = None

def _build_game_file_cache() -> dict[str, str]:
    """Scan original snapshot files and build ep_id → game_file mapping."""
    cache: dict[str, str] = {}
    for path in sorted(SNAPSHOTS_DIR.rglob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        orig_id = snap.get("id", "unknown")
        model_clean = snap.get("model", "").replace("/", "_")
        parts = path.stem.split("_")
        inst_num = parts[1] if len(parts) >= 2 else "000"
        uid = f"{orig_id}_{model_clean}_inst{inst_num}"
        gf = snap.get("game_file", "")
        if gf:
            cache[uid] = gf
    return cache


def get_game_file(ep_id: str) -> str:
    """Return the game_file path for a dataset episode ID."""
    global _game_file_cache
    if _game_file_cache is None:
        _game_file_cache = _build_game_file_cache()
    gf = _game_file_cache.get(ep_id, "")
    if not gf:
        raise ValueError(f"game_file not found for episode ID: {ep_id!r}")
    return gf

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snapshots(task_short: str = None) -> dict[str, dict]:
    """Load episodes indexed by episode ID.
    If task_short is None, loads all task types (needed for Type-3 cross-task snapshots).
    """
    task_shorts_to_load = TASK_SHORT_LIST if task_short is None else [task_short]
    result: dict[str, dict] = {}
    for ts in task_shorts_to_load:
        path = DATASET_ROOT / ts / "snapshots.json"
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


def parse_action(response: str) -> str:
    """Extract ACTION from LLM response."""
    if not response:
        return "look"
    for line in response.strip().split("\n"):
        m = re.search(r"ACTION:\s*(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return "look"

# ---------------------------------------------------------------------------
# Game state / episode helpers
# ---------------------------------------------------------------------------

def restore_game_state(env: AlfWorldEnv,
                       trajectory: list[dict],
                       decision_point: int):
    """
    Replay trajectory to reach the state at decision_point.

    AlfWorld trajectory convention:
      step 0: initial obs from reset(), action=None
      step i (i>0): obs RECEIVED after taking step[i]["action"]

    To arrive at trajectory[decision_point]["obs"], execute actions at
    trajectory[1]["action"] through trajectory[decision_point]["action"].

    This differs from Jericho (where trajectory[0] already has an action
    and we replay trajectory[:decision_point]).
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


def run_episode_from_decision_point(
    env: AlfWorldEnv,
    agent_fn,
    initial_ob: str,
    initial_info: dict,
    decision_point: int,
    log_path: Path,
) -> list[dict]:
    """Run from decision_point using agent_fn. Returns trajectory."""
    trajectory = []
    ob, info = initial_ob, initial_info

    with open(log_path, "a", encoding="utf-8") as f:
        for step in range(decision_point, STEP_LIMIT):
            action, raw_response = agent_fn(ob, step)

            f.write(f"[Step {step}] OBS: {ob[:120]}\n")
            f.write(f"           ACTION: {action}\n")

            trajectory.append({
                "step": step,
                "obs": ob,
                "action": action,
                "raw_response": raw_response,
            })

            ob, done, info = env.step(action)
            trajectory[-1]["progress_after"] = info.get("progress", 0.0)
            trajectory[-1]["progress_strict_after"] = info.get("progress_strict", 0.0)
            trajectory[-1]["progress_lenient_after"] = info.get("progress_lenient", 0.0)
            trajectory[-1]["won"] = info.get("won", False)

            print(
                f"  [step {step}] {action[:30]:30s} "
                f"→ progress={info.get('progress', 0):.3f}, won={info.get('won', False)}"
            )

            if done:
                break

    return trajectory
