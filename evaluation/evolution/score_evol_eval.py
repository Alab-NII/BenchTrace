"""
score_evol_eval.py  —  universal EvolEval avoidance metric

Avoidance metric:
  Operation failure:
    Avoided = agent's post-DP trajectory never produces the same obs_after as f*.
    For text-based games (jericho/alfworld/scienceworld/babyai): cases where the
    agent never visits f*'s location are excluded (return None).
    For bundled_web_shopping: matched by subtask_idx + extracted selection letter
    rather than obs_after (which is task-structure-determined, not action-determined).

  Strategy failure:
    Default: avoided = max recall of f*'s obs over any N-step sliding window within
    f*'s where range vs agent's full post-DP obs set is below RECALL_THRESHOLD.
    alfworld: progress-based — avoided if agent achieves strictly better progress
    than the original failed episode (obs-recall breaks because where covers ~38/45
    steps and evotest's correct navigation visits the same rooms as the failure).
    bundled_web_shopping: same subtask_idx + selection matching as operation.

Usage:
  python score_evol_eval.py --task jericho \\
      [--results_dir main_result/Jericho] \\
      [--dataset_path final_dataset/jericho/all.json] \\
      [--subtasks balances detective library temple zork1 zork3] \\
      [--baselines naive react reflexion rag evotest ...]

  python score_evol_eval.py --task babyai \\
      [--results_dir main_result/BabyAI] \\
      [--subtasks level-1 level-2 ... level-10]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RECALL_WINDOW   = 10     # strategy: sliding window size
RECALL_THRESHOLD = 0.6   # strategy: not-avoided if max_recall >= this

# Tasks where room-based location extraction applies
TEXT_ADVENTURE_TASKS = {"jericho", "alfworld", "scienceworld", "babyai"}

# Tasks using progress-based strategy avoidance (obs-recall is unreliable)
PROGRESS_STRATEGY_TASKS = {"alfworld"}

# Tasks using subtask_idx + selection matching instead of obs_after
SUBTASK_SELECTION_TASKS = {"bundled_web_shopping"}

# Tasks where max_recall=0.0 strategy cases count as avoided (no trivial exclusion)
NO_TRIVIAL_EXCL_TASKS = {"babyai"}

# Default subtasks per task
DEFAULT_SUBTASKS = {
    "jericho":              ["balances", "detective", "library", "temple", "zork1", "zork3"],
    "babyai":               [f"level-{i}" for i in range(1, 11)],
    "alfworld":             ["pick_and_place", "look_at_obj", "pick_clean",
                             "pick_heat", "pick_cool", "pick_two"],
    "scienceworld":         ["boil", "chemistry-mix", "find-living-thing", "grow-plant",
                             "inclined-plane-friction-named-surfaces",
                             "measure-melting-point-known-substance",
                             "melt", "power-component",
                             "test-conductivity-of-unknown-substances",
                             "use-thermometer"],
    "bundled_web_shopping": ["baking", "beauty", "electronics", "grocery",
                             "home", "office", "pet", "sports", "toys"],
    "group_travel_planning":["5_travelers", "6_travelers", "7_travelers", "8_travelers"],
}

DEFAULT_BASELINES = ["naive", "non_evolution", "react", "reflexion",
                     "rag", "remem", "memrl", "autoskill", "evotest"]

TYPES     = [1, 2, 3]
DISTANCES = [1, 5, 10]

# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _is_room_header(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) < 60 and not line.lower().startswith("you ")


def extract_location(trajectory: list[dict], idx: int) -> str:
    for i in range(idx, -1, -1):
        first = trajectory[i]["obs"].strip().split("\n")[0].strip()
        if _is_room_header(first):
            return first.lower()
    return "unknown"


def get_obs_after(trajectory: list[dict], idx: int) -> str | None:
    if idx + 1 < len(trajectory):
        return trajectory[idx + 1]["obs"].strip().lower()
    return None


def action_verb(trajectory: list[dict], idx: int) -> str:
    action = trajectory[idx].get("action", "").strip().lower()
    return action.split()[0] if action else ""


def reproduces_failure(orig_traj: list[dict], orig_idx: int,
                       post_traj: list[dict], post_idx: int) -> bool:
    f_obs = get_obs_after(orig_traj, orig_idx)
    a_obs = get_obs_after(post_traj, post_idx)
    if f_obs is None or a_obs is None:
        return action_verb(orig_traj, orig_idx) == action_verb(post_traj, post_idx)
    return f_obs == a_obs


# ---------------------------------------------------------------------------
# BundledWebShopping: subtask_idx + selection matching
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 extended thinking)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_sel_letter(action: str) -> str:
    """Extract the selection letter (a-e) from action text."""
    if not action:
        return ""
    action = _strip_thinking(action)
    m = re.search(r"Selection:\s*\[([A-Ea-e])\]", action, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\[([A-Ea-e])\]", action)
    if m:
        return m.group(1).lower()
    return ""


def _parse_options(obs: str) -> dict[str, str]:
    """Extract {letter: product_name} from decision-point obs."""
    opts = {}
    for m in re.finditer(r"\[([A-Ea-e])\]\s+(.+)", obs):
        opts[m.group(1).lower()] = m.group(2).strip().lower()
    return opts


def reproduces_failure_bundled(orig_traj: list[dict], orig_idx: int,
                               post_traj: list[dict],
                               dp_obs: str = "") -> bool | None:
    """Check if agent selects the same product at the same subtask_idx.

    Compares by product name (from the obs option listing) rather than
    selection letter, so cross-episode letter remapping doesn't matter.
    Returns True (same product), False (different product), or None
    (subtask not reached, or product can't be determined).
    """
    f_subtask = orig_traj[orig_idx].get("subtask_idx")
    if f_subtask is None:
        return None

    opts = _parse_options(dp_obs)
    f_let = _extract_sel_letter(orig_traj[orig_idx].get("action", ""))
    f_prod = opts.get(f_let, "") if f_let else ""
    if not f_prod:
        return None  # can't determine original product — exclude

    for step in post_traj:
        if step.get("subtask_idx") == f_subtask:
            a_let = _extract_sel_letter(step.get("action", ""))
            a_prod = opts.get(a_let, "") if a_let else ""
            if not a_prod:
                return None  # can't determine agent product — exclude
            return a_prod == f_prod
    return None  # subtask never reached — exclude


# ---------------------------------------------------------------------------
# Strategy: recall-based sliding window
# ---------------------------------------------------------------------------

def _obs_set(traj: list[dict], start: int, end: int) -> set[str]:
    return {traj[i]["obs"].strip().lower()
            for i in range(start, min(end + 1, len(traj)))}


def max_recall_over_windows(orig_traj: list[dict], where: list[int],
                            agent_obs_set: set[str], N: int) -> float:
    best = 0.0
    w_start, w_end = where[0], where[1]
    for w in range(w_start, min(w_end + 1, len(orig_traj) - N + 1)):
        f_obs = _obs_set(orig_traj, w, w + N - 1)
        if not f_obs:
            continue
        recall = len(f_obs & agent_obs_set) / len(f_obs)
        if recall > best:
            best = recall
    return best


# ---------------------------------------------------------------------------
# Per-result avoidance
# ---------------------------------------------------------------------------

def compute_avoidance(result: dict, episode_index: dict[str, dict],
                      use_location: bool, task: str = "") -> float | None:
    ep_id = result.get("test_episode_id")
    if not ep_id or ep_id not in episode_index:
        return None

    ep = episode_index[ep_id]
    orig_traj = ep["snapshot"]["trajectory"]
    post_dp_traj = result.get("trajectory_from_decision_point", [])
    if not post_dp_traj:
        return None

    fi = result["target_failure_instance"]
    cat = fi["type"].split("/")[0]
    where = fi["where"]

    # ── BundledWebShopping: subtask_idx + product-name matching ─────────────
    if task in SUBTASK_SELECTION_TASKS:
        dp = result.get("decision_point")
        dp_obs = orig_traj[dp]["obs"] if dp is not None and dp < len(orig_traj) else ""
        for orig_idx in range(where[0], min(where[1] + 1, len(orig_traj))):
            match = reproduces_failure_bundled(orig_traj, orig_idx, post_dp_traj, dp_obs)
            if match is True:
                return 0.0
            if match is False:
                return 1.0
        return None  # subtask never reached in post-DP

    # ── Standard operation failure ────────────────────────────────────────────
    if cat == "operation":
        if use_location:
            f_loc = extract_location(orig_traj, where[0])
            agent_locs = {extract_location(post_dp_traj, i)
                          for i in range(len(post_dp_traj))}
            if f_loc not in agent_locs:
                return None  # never visited — exclude

        avoided = True
        for orig_idx in range(where[0], min(where[1] + 1, len(orig_traj))):
            for post_idx in range(len(post_dp_traj)):
                if reproduces_failure(orig_traj, orig_idx, post_dp_traj, post_idx):
                    avoided = False
                    break
            if not avoided:
                break
        return 1.0 if avoided else 0.0

    # ── Strategy failure ──────────────────────────────────────────────────────
    else:
        # AlfWorld: obs-recall breaks because where spans ~38/45 steps and
        # better agents (evotest) visit the same correct rooms, inflating recall.
        # Use progress-based: avoided if agent beats the original episode's progress.
        # Exclude orig=0 episodes (complete failures) — any agent trivially beats them.
        if task in PROGRESS_STRATEGY_TASKS:
            snap = ep["snapshot"]
            max_score = snap.get("max_score") or 1.0
            orig_progress = snap["final_score"] / max_score
            if orig_progress == 0.0:
                return None  # trivially beaten by any agent — exclude
            agent_progress = result.get("final_progress", 0.0)
            return 1.0 if agent_progress > orig_progress else 0.0

        agent_obs = {step["obs"].strip().lower() for step in post_dp_traj}
        max_r = max_recall_over_windows(orig_traj, where, agent_obs, RECALL_WINDOW)
        if max_r == 0.0 and task not in NO_TRIVIAL_EXCL_TASKS:
            return None  # agent never visited failure region — exclude (trivially avoided)
        return 0.0 if max_r >= RECALL_THRESHOLD else 1.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path) -> dict[str, dict]:
    with open(dataset_path) as f:
        data = json.load(f)
    return {ep["id"]: ep for ep in data["episodes"]}


def load_results(results_dir: Path, subtasks: list[str],
                 baselines: list[str]) -> dict[str, list[dict]]:
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    for subtask in subtasks:
        for baseline in baselines:
            path = results_dir / subtask / baseline / "results.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            by_baseline[baseline].extend(data["results"])
    return dict(by_baseline)


# ---------------------------------------------------------------------------
# Aggregation & display
# ---------------------------------------------------------------------------

Cell = tuple[int, int]


def aggregate(results: list[dict], episode_index: dict[str, dict],
              use_location: bool, task: str = "") -> dict[Cell, dict]:
    cells: dict[Cell, list[float]] = defaultdict(list)
    for r in results:
        if "type" not in r or "distance" not in r:
            continue
        t, d = r["type"], r["distance"]
        av = compute_avoidance(r, episode_index, use_location, task)
        if av is not None:
            cells[(t, d)].append(av)
    return {
        cell: {"avoidance_rate": sum(vs) / len(vs), "n": len(vs)}
        for cell, vs in cells.items()
    }


def print_table(by_baseline: dict[str, dict[Cell, dict]]):
    distances = sorted({d for agg in by_baseline.values() for _, d in agg})
    types     = sorted({t for agg in by_baseline.values() for t, _ in agg})

    header_cells = [f"T{t}/D{d}" for t in types for d in distances]
    col_w, bl_w = 12, 16
    print(f"\nAvoidance Rate")
    print(f"{'Baseline':<{bl_w}}", end="")
    for t in types:
        for d in distances:
            print(f"  T{t}/D{d:>2}", end="")
    print(f"  {'Mean':>6}")
    print("-" * (bl_w + col_w * len(header_cells) + 8))

    for baseline, agg in sorted(by_baseline.items()):
        print(f"{baseline:<{bl_w}}", end="")
        vals = []
        for t in types:
            for d in distances:
                cell = agg.get((t, d))
                if cell:
                    v = cell["avoidance_rate"]
                    vals.append(v)
                    print(f"  {v:>6.3f}", end="")
                else:
                    print(f"  {'—':>6}", end="")
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {mean:>6.3f}")


def print_by_subtask(results: list[dict], episode_index: dict[str, dict],
                     use_location: bool, baseline: str, task: str = ""):
    by_sub: dict[str, list[float]] = defaultdict(list)
    for r in results:
        av = compute_avoidance(r, episode_index, use_location, task)
        game_field = r.get("game") or r.get("task_type") or r.get("level") or "?"
        if av is not None:
            by_sub[game_field].append(av)
    print(f"\n  Per-subtask avoidance ({baseline}):")
    for sub in sorted(by_sub):
        vs = by_sub[sub]
        print(f"    {sub}: {sum(vs)/len(vs):.3f}  (n={len(vs)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description="Score EvolEval avoidance rate (all tasks).")
    p.add_argument("--task", required=True,
                   choices=["jericho", "babyai", "alfworld", "scienceworld",
                             "bundled_web_shopping", "group_travel_planning"],
                   help="Task name")
    p.add_argument("--results_dir", default=None,
                   help="Root results directory (default: main_result/<Task>)")
    p.add_argument("--dataset_path", default=None,
                   help="Path to all.json (default: final_dataset/<task>/all.json)")
    p.add_argument("--subtasks", nargs="+", default=None,
                   help="Subtask names (games/levels/task_types). Default: all.")
    p.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    p.add_argument("--per_subtask", action="store_true",
                   help="Print per-subtask breakdown per baseline")
    p.add_argument("--output", default=None, help="Optional JSON output path")
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).parent
    task = args.task

    # Resolve paths
    task_dir_map = {
        "jericho":               "Jericho",
        "babyai":                "BabyAI",
        "alfworld":              "AlfWorld",
        "scienceworld":          "ScienceWorld",
        "bundled_web_shopping":  "BundledWebShopping",
        "group_travel_planning": "GroupTravelPlanning",
    }
    dataset_dir_map = {
        "jericho":               "jericho",
        "babyai":                "babyai",
        "alfworld":              "alfworld",
        "scienceworld":          "scienceworld",
        "bundled_web_shopping":  "bundled_web_shopping",
        "group_travel_planning": "group_travel_planning",
    }

    results_dir = Path(args.results_dir) if args.results_dir else \
                  here / "main_result" / task_dir_map[task]
    dataset_path = Path(args.dataset_path) if args.dataset_path else \
                   here / "final_dataset" / dataset_dir_map[task] / "all.json"
    subtasks = args.subtasks or DEFAULT_SUBTASKS[task]
    use_location = task in TEXT_ADVENTURE_TASKS

    strat_note = ""
    if task in PROGRESS_STRATEGY_TASKS:
        strat_note = "  [strategy: progress-based]"
    elif task in SUBTASK_SELECTION_TASKS:
        strat_note = "  [all: subtask_idx+selection]"

    print(f"Task: {task}  (location_check={use_location}){strat_note}")
    print(f"Loading dataset from {dataset_path} ...")
    episode_index = load_dataset(dataset_path)
    print(f"  Loaded {len(episode_index)} episodes.")

    print(f"Loading results from {results_dir} ...")
    by_baseline = load_results(results_dir, subtasks, args.baselines)
    total = sum(len(rs) for rs in by_baseline.values())
    print(f"  Loaded {total} results across {len(by_baseline)} baselines.")

    agg_by_baseline: dict[str, dict[Cell, dict]] = {}
    for baseline, results in sorted(by_baseline.items()):
        agg_by_baseline[baseline] = aggregate(results, episode_index, use_location, task)
        if args.per_subtask:
            print_by_subtask(results, episode_index, use_location, baseline, task)

    print_table(agg_by_baseline)

    if args.output:
        out = {
            bl: {f"T{t}_D{d}": v for (t, d), v in agg.items()}
            for bl, agg in agg_by_baseline.items()
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
