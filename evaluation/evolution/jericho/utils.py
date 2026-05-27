"""
Shared constants, prompts, and helper functions for EvolEval runners.
"""

import sys
import json
import re
from pathlib import Path

EVOTEST_PATH = Path(__file__).parent.parent / "EvoTest"
sys.path.insert(0, str(EVOTEST_PATH))
from src.env import JerichoEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text
from src.utils import game_file  # noqa: F401  (re-exported for runner convenience)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).parent.parent.parent / "final_dataset" / "jericho"
ROM_DIR = EVOTEST_PATH / "jericho-games"
RESULTS_ROOT = Path(__file__).parent / "results"

MAX_CONTEXT_TOKENS = 32000
STEP_LIMIT = 110

GAMES = ["balances", "detective", "library", "temple", "zork1", "zork3"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert player aiming to complete a text-based adventure game. "
    "Points are given for making progress in the game. "
    "Study the provided game history carefully and use it to make better decisions."
)

# For baselines that use evolution context (with reasoning)
ACTION_FORMAT_INSTRUCTION = """
Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

Common actions include: look, inventory, directions (north, south, east, west, up, down, etc.), \
examine X, get X, drop X, open X, enter X, give X to Y, and other context-specific commands. \
Any X or Y MUST be a *SINGLE WORD*. Do NOT repeat the same failed action multiple times.
"""

# For non-evolution baseline (no reasoning needed)
NON_EVO_ACTION_FORMAT = """
Your response MUST strictly follow this format and include nothing else:
ACTION: [short word or phrase for text command to execute]

Common actions include: look, inventory, directions (north, south, east, west, up, down, etc.), \
examine X, get X, drop X, open X, enter X, give X to Y, and other context-specific commands. \
Any X or Y MUST be a *SINGLE WORD*. Do NOT repeat the same failed action multiple times.
"""

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snapshots(game: str = None) -> dict[str, dict]:
    """Load episodes indexed by episode ID.
    If game is None, loads all games (needed for T3 cross-game evolution snapshots).
    """
    games_to_load = GAMES if game is None else [game]
    result = {}
    for g in games_to_load:
        path = DATASET_ROOT / g / "snapshots.json"
        with open(path) as f:
            data = json.load(f)
        result.update({ep["id"]: ep for ep in data["episodes"]})
    return result

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_trajectory(trajectory: list[dict], header: str = None) -> str:
    """Format a list of trajectory steps as readable text."""
    lines = []
    if header:
        lines.append(f"=== {header} ===")
    for step in trajectory:
        lines.append(f"[Step {step['step']}] Observation: {step['obs']}")
        if step.get("inv"):
            lines.append(f"           Inventory: {step['inv']}")
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

def restore_game_state(env: JerichoEnv, trajectory: list[dict], decision_point: int):
    """Replay trajectory up to decision_point to restore game state."""
    ob, info = env.reset()
    for step_data in trajectory[:decision_point]:
        ob, _, done, info = env.step(step_data["action"])
        if done:
            break
    return ob, info


def run_episode_from_decision_point(
    env: JerichoEnv,
    agent_fn,
    initial_ob: str,
    initial_info: dict,
    decision_point: int,
    log_path: Path,
) -> list[dict]:
    """Run a full episode from decision_point using agent_fn. Returns trajectory."""
    trajectory = []
    ob, info = initial_ob, initial_info

    with open(log_path, "a", encoding="utf-8") as f:
        for step in range(decision_point, STEP_LIMIT):
            action, raw_response = agent_fn(ob, info.get("inv", ""), step)

            f.write(f"[Step {step}] OBS: {ob[:120]}\n")
            f.write(f"           ACTION: {action}\n")

            trajectory.append({
                "step": step,
                "obs": ob,
                "inv": info.get("inv", ""),
                "action": action,
                "raw_response": raw_response,
            })

            ob, reward, done, info = env.step(action)
            trajectory[-1]["reward"] = reward
            trajectory[-1]["score_after"] = info.get("score", 0)

            print(f"  [step {step}] {action[:30]:30s} → reward={reward}, score={info.get('score',0)}")

            if done:
                break

    return trajectory
