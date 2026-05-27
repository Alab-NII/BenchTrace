#!/usr/bin/env python3
"""
Assemble Evolution Evaluation dataset for Bundled Web Shopping.

Mirrors the Jericho/AlfWorld/BabyAI/ScienceWorld assembler, adapted for
Bundled Web Shopping (closed-world product selection tasks).

Key differences from step-based environments (AlfWorld, ScienceWorld, etc.):
  - Each "step" = one subtask (product selection question)
  - Trajectory uses paired steps: even step = obs before action (action=None),
    odd step = obs after action (action present, correct=True/False)
  - "where" = [step_idx, step_idx] where step_idx is the ODD (action) step
    of the failing subtask; always odd (1, 3, 5, ...)
  - decision_point = where[0] - 1  →  the even obs step of the failing subtask
  - SAME_INSTANCE_STEP_THRESHOLD = 2: failures within ±1 subtask (±2 steps)
    count as the same instance (Type 1); larger gaps → Type 2
  - system/output_truncation excluded: model cut-off is a system issue, not
    a learnable strategy failure
  - Env needs: BundledShoppingEnv(task_id, dataset) — task_id encoded in ep_id

3D task matrix:
  Type 1: same category, same failure type, similar where[0] (≤2 steps)
  Type 2: same category, same failure type, different subtask (>2 steps)
  Type 3: different category, same failure type
  Distance: 1, 2, 3, 5, 10
  FailureType: one cell per meaningful failure type

Output: final_dataset/bundled_web_shopping/{game}/evolution_evaluation.json
"""

import json
import random
from pathlib import Path
from typing import Optional

DATASET_ROOT = Path(__file__).parent.parent.parent / "final_dataset" / "bundled_web_shopping"
GAME_LIST = ["baking", "beauty", "electronics", "grocery", "home"]

DISTANCES = [1, 2, 3, 5, 10]
TYPES = [1, 2, 3]
MAX_PER_CELL = 1

# ±2 steps = same or adjacent subtask → "same instance"
# Each subtask occupies 2 steps, so threshold=2 means within ±1 subtask.
SAME_INSTANCE_STEP_THRESHOLD = 2

# Exclude system-level failures: output_truncation is a model capacity issue,
# not a strategy/operation mistake an agent can learn to avoid from examples.
EXCLUDED_FAILURE_TYPES = {"system/format", "unknown", "system/output_truncation"}

random.seed(42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_snapshots() -> dict[str, list[dict]]:
    """Returns {game: [episode, ...]}."""
    all_snapshots: dict[str, list[dict]] = {}
    for game in GAME_LIST:
        path = DATASET_ROOT / game / "snapshots.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        all_snapshots[game] = data["episodes"]
    return all_snapshots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scoring_method(failure_type: str) -> str:
    """
    operation/* → one_step: evaluate only the failing subtask (one selection).
    strategy/* → full_episode: the cascade/propagation failures affect multiple
                 subtasks; evaluate the complete episode.
    """
    return "one_step" if failure_type.startswith("operation/") else "full_episode"


def has_failure_type(episode: dict, failure_type: str) -> bool:
    fi = episode["failure_instances"]
    all_failures = fi.get("core_failure", []) + fi.get("marginal_failure", [])
    return any(f["type"] == failure_type for f in all_failures)


def is_same_instance(fi_a: dict, fi_b: dict) -> bool:
    """True if both failures occur within ±THRESHOLD steps (±1 subtask)."""
    return (
        fi_a["type"] == fi_b["type"]
        and abs(fi_a["where"][0] - fi_b["where"][0]) <= SAME_INSTANCE_STEP_THRESHOLD
    )


# ---------------------------------------------------------------------------
# Key evolution snapshot finders
# ---------------------------------------------------------------------------

def find_key_type1(target_fi: dict, test_ep_id: str,
                   game_episodes: list[dict]) -> Optional[dict]:
    candidates = [
        ep for ep in game_episodes
        if ep["id"] != test_ep_id
        and any(
            is_same_instance(fi, target_fi)
            for fi in ep["failure_instances"].get("core_failure", [])
        )
    ]
    return random.choice(candidates) if candidates else None


def find_key_type2(target_fi: dict, test_ep_id: str,
                   game_episodes: list[dict]) -> Optional[dict]:
    candidates = [
        ep for ep in game_episodes
        if ep["id"] != test_ep_id
        and any(
            fi["type"] == target_fi["type"] and not is_same_instance(fi, target_fi)
            for fi in ep["failure_instances"].get("core_failure", [])
        )
    ]
    return random.choice(candidates) if candidates else None


def find_key_type3(target_fi: dict, target_game: str,
                   all_snapshots: dict[str, list[dict]]) -> Optional[dict]:
    candidates = [
        ep
        for game, episodes in all_snapshots.items()
        if game != target_game
        for ep in episodes
        if any(
            fi["type"] == target_fi["type"]
            for fi in ep["failure_instances"].get("core_failure", [])
        )
    ]
    return random.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Gap snapshot finder
# ---------------------------------------------------------------------------

def find_gaps(n_gaps: int, target_fi_type: str, exclude_ids: set[str],
              game_episodes: list[dict],
              all_snapshots: dict[str, list[dict]],
              task_type: int) -> list[str]:
    if n_gaps == 0:
        return []

    if task_type in [1, 2]:
        pool = [
            ep for ep in game_episodes
            if ep["id"] not in exclude_ids
            and not has_failure_type(ep, target_fi_type)
        ]
    else:
        pool = [
            ep
            for episodes in all_snapshots.values()
            for ep in episodes
            if ep["id"] not in exclude_ids
            and not has_failure_type(ep, target_fi_type)
        ]

    if not pool:
        return []
    if len(pool) >= n_gaps:
        return [ep["id"] for ep in random.sample(pool, n_gaps)]
    chosen = (pool * ((n_gaps // len(pool)) + 1))[:n_gaps]
    return [ep["id"] for ep in chosen]


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def assemble_evolution_evaluation(
    all_snapshots: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    results: dict[str, list[dict]] = {g: [] for g in all_snapshots}
    cell_counts: dict[str, dict] = {g: {} for g in all_snapshots}

    for game in all_snapshots:
        game_episodes = all_snapshots[game]
        cell_candidates: dict[tuple, list[dict]] = {}

        for ep in game_episodes:
            ep_id = ep["id"]
            core_failures = ep["failure_instances"].get("core_failure", [])

            for fi in core_failures:
                if fi["type"] in EXCLUDED_FAILURE_TYPES:
                    continue
                decision_point = fi["where"][0] - 1
                if decision_point < 0:
                    continue

                eval_method = scoring_method(fi["type"])

                for task_type in TYPES:
                    for distance in DISTANCES:
                        cell_key = (task_type, distance, fi["type"])
                        n_gaps = distance - 1

                        if task_type == 1:
                            key_ep = find_key_type1(fi, ep_id, game_episodes)
                        elif task_type == 2:
                            key_ep = find_key_type2(fi, ep_id, game_episodes)
                        else:
                            key_ep = find_key_type3(fi, game, all_snapshots)

                        if key_ep is None:
                            continue

                        exclude_ids = {ep_id, key_ep["id"]}
                        gap_ids = find_gaps(
                            n_gaps, fi["type"], exclude_ids,
                            game_episodes, all_snapshots, task_type,
                        )
                        if len(gap_ids) < n_gaps:
                            continue

                        candidate = {
                            "game": game,
                            "type": task_type,
                            "distance": distance,
                            "target_failure_instance": {
                                "type": fi["type"],
                                "where": fi["where"],
                                "diagnosis": fi.get("diagnosis", ""),
                            },
                            "test_snapshot": {
                                "episode_id": ep_id,
                                "decision_point": decision_point,
                            },
                            "evolution_snapshots": [key_ep["id"]] + gap_ids,
                            "scoring_method": eval_method,
                        }
                        cell_candidates.setdefault(cell_key, []).append(candidate)

        # Sample up to MAX_PER_CELL per cell; prefer unique test episodes
        task_idx = 0
        for cell_key in sorted(cell_candidates):
            candidates = cell_candidates[cell_key]
            random.shuffle(candidates)

            selected = []
            used_test_episodes: set[str] = set()
            remainder = []
            for c in candidates:
                ep_id = c["test_snapshot"]["episode_id"]
                if ep_id not in used_test_episodes:
                    selected.append(c)
                    used_test_episodes.add(ep_id)
                else:
                    remainder.append(c)
                if len(selected) >= MAX_PER_CELL:
                    break

            if len(selected) < MAX_PER_CELL:
                for c in remainder:
                    selected.append(c)
                    if len(selected) >= MAX_PER_CELL:
                        break

            task_type_dim, distance, failure_type = cell_key
            ft_slug = failure_type.replace("/", "_")
            game_slug = game  # no hyphens in category names
            for c in selected:
                c["id"] = (
                    f"ee_{game_slug}_t{task_type_dim}"
                    f"_d{distance}_{ft_slug}_{task_idx:04d}"
                )
                results[game].append(c)
                task_idx += 1
            cell_counts[game][cell_key] = len(selected)

    return results, cell_counts


def main():
    print("Loading snapshots...")
    all_snapshots = load_all_snapshots()
    for game, eps in all_snapshots.items():
        print(f"  {game}: {len(eps)} episodes")

    print("\nAssembling Evolution Evaluation tasks...")
    results, cell_counts = assemble_evolution_evaluation(all_snapshots)

    print("\n=== Results ===")
    total = 0
    for game in GAME_LIST:
        if game not in results:
            continue
        tasks = results[game]
        total += len(tasks)
        out_path = DATASET_ROOT / game / "evolution_evaluation.json"
        output = {
            "game": game,
            "n_tasks": len(tasks),
            "cell_counts": {
                f"type{k[0]}_dist{k[1]}_{k[2].replace('/', '_')}": v
                for k, v in sorted(cell_counts[game].items())
            },
            "tasks": tasks,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n{game}: {len(tasks)} tasks → {out_path}")
        for k in sorted(cell_counts[game]):
            print(
                f"  Type {k[0]}, Dist {k[1]:2d}, "
                f"{k[2]:50s}: {cell_counts[game][k]}"
            )

    print(f"\nTotal: {total} tasks across {len(results)} categories")


if __name__ == "__main__":
    main()
