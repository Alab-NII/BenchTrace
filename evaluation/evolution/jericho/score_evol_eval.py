"""
score_evol_eval.py

Computes avoidance rate for Evolution Evaluation results (Jericho / JTTL).

Avoidance metric (right column in Table 3):
  - Operation failure: for each step in post-DP, we check whether it "reproduces" f*.
    Two steps are the same action if:
      - obs_after (next step's obs) is NOT a generic error message → match by obs_after equality
      - obs_after IS a generic error message → fall back to (verb + location) matching
    Avoided = agent never reproduces f*'s action anywhere in post-DP trajectory.
  - Strategy failure: avoided if cosine similarity between action-frequency vectors of
    f*.where and the post-DP trajectory < COSINE_THRESHOLD (default 0.5).

Usage:
    conda run -n Fraud python score_evol_eval.py \\
        [--results_dir ../../main_result/Jericho] \\
        [--dataset_dir ../../final_dataset/jericho] \\
        [--baselines naive react reflexion rag evotest ...] \\
        [--games balances detective library temple zork1 zork3]
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

COSINE_THRESHOLD = 0.5   # strategy failure: cosine(action freq) >= threshold → not avoided

# Generic parser error messages that carry no specific action identity
GENERIC_OBS = frozenset({
    "you can't see any such thing.",
    "that's not a verb i recognise.",
    "you can't go that way.",
    "i didn't understand that sentence.",
    "i only understood you as far as wanting to look.",
    "you can only do that to something animate.",
    "there was no verb in that sentence!",
    "you can't see any figure here.",
})

# Navigation verbs: obs_after depends on game state, not action identity → verb+location fallback
NAV_VERBS = frozenset({
    "north", "south", "east", "west", "up", "down",
    "ne", "nw", "se", "sw", "go", "enter", "exit",
    "climb", "leave", "out", "in",
})

GAMES    = ["balances", "detective", "library", "temple", "zork1", "zork3"]
TYPES    = [1, 2, 3]
DISTANCES = [1, 5, 10]

# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------

def _is_room_header(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) < 60 and not line.lower().startswith("you ")


def extract_location(trajectory: list[dict], list_idx: int) -> str:
    """Current room at list_idx: scan backwards for first room-header obs."""
    for i in range(list_idx, -1, -1):
        first = trajectory[i]["obs"].strip().split("\n")[0].strip()
        if _is_room_header(first):
            return first.lower()
    return "unknown"


def get_obs_after(trajectory: list[dict], idx: int) -> str | None:
    """Observation produced by action at idx (= trajectory[idx+1].obs), or None."""
    if idx + 1 < len(trajectory):
        return trajectory[idx + 1]["obs"].strip().lower()
    return None


def action_verb(trajectory: list[dict], idx: int) -> str:
    """First word of action at idx."""
    action = trajectory[idx]["action"].strip().lower()
    return action.split()[0] if action else ""


def reproduces_failure(orig_traj: list[dict], orig_idx: int,
                       post_traj: list[dict], post_idx: int) -> bool:
    """True if action at post_idx reproduces the failure action at orig_idx.

    Matching rule: obs_after equality only.
    If obs_after is unavailable for either side, fall back to verb match.
    """
    f_obs = get_obs_after(orig_traj, orig_idx)
    a_obs = get_obs_after(post_traj, post_idx)
    if f_obs is None or a_obs is None:
        return action_verb(orig_traj, orig_idx) == action_verb(post_traj, post_idx)
    return f_obs == a_obs


def action_freq(trajectory: list[dict], start_idx: int, end_idx: int) -> Counter:
    """Action frequency counter for list indices [start_idx, end_idx]."""
    return Counter(
        trajectory[i]["action"].strip().lower()
        for i in range(start_idx, min(end_idx + 1, len(trajectory)))
    )


# ---------------------------------------------------------------------------
# Avoidance per result
# ---------------------------------------------------------------------------

def cosine(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def compute_avoidance(result: dict, episode_index: dict[str, dict]) -> float | None:
    """Return 1.0 (avoided), 0.0 (not avoided), or None (missing data)."""
    ep_id = result.get("test_episode_id")
    if not ep_id or ep_id not in episode_index:
        return None

    ep = episode_index[ep_id]
    orig_traj = ep["snapshot"]["trajectory"]
    post_dp_traj = result.get("trajectory_from_decision_point", [])
    if not post_dp_traj:
        return None

    fi = result["target_failure_instance"]
    failure_category = fi["type"].split("/")[0]   # "operation" or "strategy"
    where = fi["where"]                            # [start_step, end_step]

    if failure_category == "operation":
        # Exclude cases where agent never visits f*'s location (not a meaningful test)
        f_loc = extract_location(orig_traj, where[0])
        agent_locs = {extract_location(post_dp_traj, i) for i in range(len(post_dp_traj))}
        if f_loc not in agent_locs:
            return None

        # avoided if agent never reproduces f*'s action anywhere in post-DP
        avoided = True
        for orig_idx in range(where[0], min(where[1] + 1, len(orig_traj))):
            for post_idx in range(len(post_dp_traj)):
                if reproduces_failure(orig_traj, orig_idx, post_dp_traj, post_idx):
                    avoided = False
                    break
            if not avoided:
                break
    else:
        # strategy: cosine similarity of action-frequency vectors
        f_star_freq = action_freq(orig_traj, where[0], where[1])
        agent_freq  = action_freq(post_dp_traj, 0, len(post_dp_traj) - 1)
        avoided = cosine(f_star_freq, agent_freq) < COSINE_THRESHOLD

    return 1.0 if avoided else 0.0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir: Path, games: list[str]) -> dict[str, dict]:
    """Load all episodes indexed by ID."""
    index = {}
    for game in games:
        path = dataset_dir / game / "snapshots.json"
        if not path.exists():
            print(f"  [warn] dataset not found: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        for ep in data["episodes"]:
            index[ep["id"]] = ep
    return index


def load_results(results_dir: Path, games: list[str], baselines: list[str]) -> dict[str, list[dict]]:
    """Load results grouped by baseline name."""
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    for game in games:
        for baseline in baselines:
            path = results_dir / game / baseline / "results.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            by_baseline[baseline].extend(data["results"])
    return dict(by_baseline)


# ---------------------------------------------------------------------------
# Aggregation & display
# ---------------------------------------------------------------------------

Cell = tuple[int, int]  # (type, distance)


def aggregate(results: list[dict], episode_index: dict[str, dict]) -> dict[Cell, dict]:
    """Aggregate avoidance rate per (type, distance) cell."""
    cells: dict[Cell, list[float]] = defaultdict(list)
    for r in results:
        t, d = r["type"], r["distance"]
        av = compute_avoidance(r, episode_index)
        if av is not None:
            cells[(t, d)].append(av)
    return {
        cell: {"avoidance_rate": sum(vs) / len(vs), "n": len(vs)}
        for cell, vs in cells.items()
    }


def print_table(by_baseline: dict[str, dict[Cell, dict]]):
    distances = sorted({d for agg in by_baseline.values() for _, d in agg})
    types     = sorted({t for agg in by_baseline.values() for t, _ in agg})

    # Header
    header_cells = [f"T{t}/D{d}" for t in types for d in distances]
    col_w = 12
    bl_w  = 16
    print(f"\n{'Avoidance Rate':}")
    print(f"{'Baseline':<{bl_w}}", end="")
    for t in types:
        for d in distances:
            print(f"  T{t}/D{d:>2}", end="")
    print(f"  {'Mean':>6}")
    print("-" * (bl_w + (col_w) * len(header_cells) + 8))

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


def print_by_game(results: list[dict], episode_index: dict[str, dict], baseline: str):
    by_game: dict[str, list[float]] = defaultdict(list)
    for r in results:
        av = compute_avoidance(r, episode_index)
        if av is not None:
            by_game[r["game"]].append(av)
    print(f"\n  Per-game avoidance ({baseline}):")
    for game in sorted(by_game):
        vs = by_game[game]
        print(f"    {game}: {sum(vs)/len(vs):.3f}  (n={len(vs)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Score EvolEval avoidance rate for JTTL/Jericho.")
    parser.add_argument("--results_dir", default=str(here.parent.parent / "main_result" / "Jericho"),
                        help="Root of main_result/Jericho (contains <game>/<baseline>/results.json)")
    parser.add_argument("--dataset_dir", default=str(here.parent.parent / "final_dataset" / "jericho"),
                        help="Root of final_dataset/jericho")
    parser.add_argument("--baselines", nargs="+",
                        default=["naive", "non_evolution", "react", "reflexion",
                                 "rag", "remem", "memrl", "autoskill", "evotest"],
                        help="Baselines to include")
    parser.add_argument("--games", nargs="+", default=GAMES, help="Games to include")
    parser.add_argument("--per_game", action="store_true", help="Print per-game breakdown per baseline")
    parser.add_argument("--output", default=None, help="Optional path to save JSON summary")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    dataset_dir = Path(args.dataset_dir)

    print(f"Loading dataset from {dataset_dir} ...")
    episode_index = load_dataset(dataset_dir, args.games)
    print(f"  Loaded {len(episode_index)} episodes.")

    print(f"Loading results from {results_dir} ...")
    by_baseline = load_results(results_dir, args.games, args.baselines)
    total = sum(len(rs) for rs in by_baseline.values())
    print(f"  Loaded {total} results across {len(by_baseline)} baselines.")

    agg_by_baseline: dict[str, dict[Cell, dict]] = {}
    for baseline, results in sorted(by_baseline.items()):
        agg_by_baseline[baseline] = aggregate(results, episode_index)
        if args.per_game:
            print_by_game(results, episode_index, baseline)

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
