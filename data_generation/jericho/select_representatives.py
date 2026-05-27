"""
select_representatives.py

For each game, select representative episodes from existing EvoTest runs
(GPT-4.1 and Qwen3-32B) that introduce at least one new semantic error.

Two models share a single error set per game (union). Episodes are processed
generation by generation (gen i = episode_i from each model).

Output: JTTL/dataset/<game>/
  - gen_<i>_<model>.json   one file per representative episode
  - index.json             summary of all representatives for this game
"""

import json
import re
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
ANNOTATE_DIR = Path(__file__).resolve().parent / "output"
DATASET_DIR  = Path(__file__).resolve().parent.parent / "dataset"
EVOTEST_DIR  = Path(__file__).resolve().parent.parent / "EvoTest" / "output"

N_EPISODES = 30

GAMES = ["detective", "library", "zork1", "zork3", "balances", "temple"]

# run directories for each (game, model) pair
RUN_DIRS = {
    ("detective", "gpt-4.1"):  "our/gpt-4.1/20260327-144758",
    ("library",   "gpt-4.1"):  "our/gpt-4.1/20260327-150958",
    ("zork1",     "gpt-4.1"):  "our/gpt-4.1/20260327-174647",
    ("zork3",     "gpt-4.1"):  "our/gpt-4.1/20260327-185009",
    ("balances",  "gpt-4.1"):  "our/gpt-4.1/20260327-200123",
    ("temple",    "gpt-4.1"):  "our/gpt-4.1/20260327-213358",
    # Qwen3 — all games same timestamp
    ("detective", "qwen3-32b"): "our/qwen3-32b/20260321-154607",
    ("library",   "qwen3-32b"): "our/qwen3-32b/20260321-154607",
    ("zork1",     "qwen3-32b"): "our/qwen3-32b/20260321-154607",
    ("zork3",     "qwen3-32b"): "our/qwen3-32b/20260321-154607",
    ("balances",  "qwen3-32b"): "our/qwen3-32b/20260321-154607",
    ("temple",    "qwen3-32b"): "our/qwen3-32b/20260321-154607",
}

ANNOTATION_SUFFIX = {
    "gpt-4.1":   "",
    "qwen3-32b": "_qwen3",
}

MODELS = ["gpt-4.1", "qwen3-32b"]

# ── Semantic key (same as plot_error_set_growth.py) ───────────────────────────

def semantic_key(game: str, error: dict) -> tuple:
    etype = error["error_type"]
    ev    = error.get("evidence", {})
    if etype == "strategy/loop":
        return (game, etype, ev.get("room", "?"), ev.get("repeated_action", "?"))
    if etype == "strategy/route_inefficiency":
        return (game, etype, ev.get("from_score"), ev.get("to_score"))
    if etype == "strategy/unexplored_destination":
        return (game, etype, ev.get("room", "?"),
                tuple(sorted(ev.get("unexplored", []))))
    if etype == "operation/feedback_blindness":
        return (game, etype, ev.get("repeated_action"))
    if etype == "operation/perception_error":
        return (game, etype, ev.get("required_room"), ev.get("correct_action"))
    if etype == "operation/decision_error":
        return (game, etype, ev.get("room"), ev.get("correct_action"))
    return (game, etype)


# ── Log parser (mirror of annotate_episodes.py) ───────────────────────────────

def parse_episode_log(log_path: Path) -> list[dict]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"={5,}", text)
    steps = []
    for block in blocks:
        step_m   = re.search(r"\[STEP\]\s*(\d+)", block)
        obs_m    = re.search(r"\[OBS\]\s*(.*?)(?=\n-{3,}|\Z)", block, re.S)
        action_m = re.search(r"\[CHOSEN_ACTION\]\s*(.+)", block)
        reward_m = re.search(r"\[REWARD\]\s*(-?\d+(?:\.\d+)?)", block)
        score_m  = re.search(r"\[CUM_REWARD\]\s*(-?\d+(?:\.\d+)?)", block)
        if not step_m:
            continue
        steps.append({
            "step":      int(step_m.group(1)),
            "obs":       obs_m.group(1).strip() if obs_m else "",
            "action":    action_m.group(1).strip() if action_m else "",
            "reward":    float(reward_m.group(1)) if reward_m else 0.0,
            "cum_score": float(score_m.group(1)) if score_m else 0.0,
        })
    return sorted(steps, key=lambda x: x["step"])


def parse_guiding_prompt(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Guiding prompt:\s*(.+?)(?:\n={3,}|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


# ── Load annotations ──────────────────────────────────────────────────────────

def load_annotations(game: str, model: str) -> dict[int, dict]:
    """Return {episode_num: annotation_dict}."""
    suffix = ANNOTATION_SUFFIX[model]
    path = ANNOTATE_DIR / f"{game}_annotations{suffix}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {ep["episode"]: ep for ep in data.get("episodes", [])}


def all_errors(ep_annotation: dict) -> list[dict]:
    errs = []
    for cat in ("system", "strategy", "operation"):
        errs.extend(ep_annotation.get("errors", {}).get(cat, []))
    return errs


# ── Main selection logic ──────────────────────────────────────────────────────

def process_game(game: str):
    out_dir = DATASET_DIR / game
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all annotations for both models
    annotations = {m: load_annotations(game, m) for m in MODELS}

    seen_errors: set[tuple] = set()
    index_entries = []

    for gen in range(N_EPISODES):
        for model in MODELS:
            ann = annotations[model].get(gen)
            if ann is None:
                continue

            errors = all_errors(ann)
            new_errors = []
            for err in errors:
                key = semantic_key(game, err)
                if key not in seen_errors:
                    seen_errors.add(key)
                    new_errors.append({**err, "semantic_key": list(key)})

            if not new_errors:
                continue  # not representative

            # Load trajectory from log file
            run_subdir = RUN_DIRS.get((game, model))
            log_file = ann.get("log_file", "")
            trajectory = []
            guiding_prompt = ""
            next_guiding_prompt = ""
            if run_subdir and log_file:
                log_path = EVOTEST_DIR / game / run_subdir / log_file
                if log_path.exists():
                    trajectory = parse_episode_log(log_path)
                    guiding_prompt = parse_guiding_prompt(log_path)

                # Next episode's guiding prompt = EvoTest reflection output
                next_ann = annotations[model].get(gen + 1)
                next_log_file = next_ann.get("log_file", "") if next_ann else ""
                if next_log_file:
                    next_log_path = EVOTEST_DIR / game / run_subdir / next_log_file
                    if next_log_path.exists():
                        next_guiding_prompt = parse_guiding_prompt(next_log_path)

            model_tag = model.replace("-", "").replace(".", "")
            episode_id = f"jericho_{game}_{model_tag}_{gen:02d}"

            snapshot = {
                # ── Identity ─────────────────────────────────────────────
                "id":             episode_id,
                "task":           "jericho",
                "game":           game,
                "generation":     gen,
                "model":          model,
                "log_file":       log_file,
                # ── Performance ──────────────────────────────────────────
                "final_score":    ann["final_score"],
                "max_score":      ann["max_score"],
                "n_steps":        ann.get("n_steps", len(trajectory)),
                # ── Rule-based annotations ────────────────────────────────
                "whether":        ann.get("whether"),
                "where":          ann.get("where"),
                "what":           ann.get("what"),
                "all_errors":     errors,
                # ── New errors introduced by this episode ─────────────────
                "new_errors":     new_errors,
                # ── Full trajectory ───────────────────────────────────────
                "guiding_prompt":      guiding_prompt,
                "trajectory":          trajectory,
                # ── EvoTest reflection (next episode's prompt, used as how template) ──
                "evotest_reflection":  next_guiding_prompt,
                # ── Golden reflection (to be filled by human annotation) ──
                "golden_reflection": {
                    "why":  None,   # rule-based template + human review
                    "how":  None,   # human annotation
                },
            }

            filename = f"gen{gen:02d}_{model_tag}.json"
            with open(out_dir / filename, "w") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            index_entries.append({
                "file":         filename,
                "generation":   gen,
                "model":        model,
                "final_score":  ann["final_score"],
                "max_score":    ann["max_score"],
                "n_steps":      ann.get("n_steps"),
                "n_new_errors": len(new_errors),
                "golden_reflection_done": False,
            })

    # Write index
    index = {
        "game":               game,
        "total_representatives": len(index_entries),
        "total_unique_errors":   len(seen_errors),
        "entries":            index_entries,
    }
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"  {game:12s}: {len(index_entries):3d} representative episodes, "
          f"{len(seen_errors):3d} unique errors")
    return index_entries


if __name__ == "__main__":
    print("Selecting representative episodes...\n")
    total_eps = 0
    for game in GAMES:
        entries = process_game(game)
        total_eps += len(entries)
    print(f"\nTotal representative episodes across all games: {total_eps}")
    print(f"Dataset written to: {DATASET_DIR}")
