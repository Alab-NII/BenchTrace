"""
reannotate_operation.py

Re-annotates operation failures in ai_annotations.json files, replacing
step ranges with a single step according to the updated localization rules:

  - operation/feedback_blindness : step(s) where the agent first repeats the
      action after receiving negative feedback for it (up to 3 per annotation)
  - operation/perception_error   : first step where the agent enters the
      correct room for the milestone
  - operation/decision_error     : step where the incorrect attempt occurs

When a game has human annotations (output/human/{game}_human.json), only
human-kept (core) failures are reannotated. Otherwise all operation failures
with a range where are processed.

Usage (dry-run to check output without saving):
    python reannotate_operation.py --game detective --dry-run

Usage (run specific games, optionally in parallel):
    python reannotate_operation.py --games detective library
    python reannotate_operation.py --all
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic

BASE   = Path(__file__).parent
OUTPUT = BASE / "output"
GAMES  = ["detective", "library", "zork1", "zork3", "balances", "temple"]

OPERATION_TYPES = {
    "operation/feedback_blindness",
    "operation/perception_error",
    "operation/decision_error",
}

IOU_THRESHOLD = 0.3

STEP_DEFINITIONS = {
    "operation/feedback_blindness": (
        "the step where the agent first repeats the same action "
        "after having already received negative feedback for it"
    ),
    "operation/perception_error": (
        "the first step where the agent enters the correct room "
        "for the milestone (even if it later leaves and re-enters)"
    ),
    "operation/decision_error": (
        "the step where the agent makes the incorrect attempt "
        "(wrong timing or wrong object)"
    ),
}

SYSTEM_PROMPT_SINGLE = """\
You are a precise failure localization assistant. You will be given:
1. A window of trajectory steps from a text-adventure episode.
2. A failure annotation describing an operation-level failure.

Your task: identify the single most accurate step number where this failure occurs,
according to the localization rule provided.

Respond with ONLY a single integer (the step number). No explanation, no JSON, no other text."""

SYSTEM_PROMPT_MULTI = """\
You are a precise failure localization assistant. You will be given:
1. A window of trajectory steps from a text-adventure episode.
2. A failure annotation describing an operation/feedback_blindness failure.

Your task: identify each distinct step where the agent first repeats an action after
receiving negative feedback for it. There may be multiple independent occurrences
(e.g. the agent ignores negative feedback for action A at step 10, and later ignores
negative feedback for a different action B at step 40 — these are two separate occurrences).

Return AT MOST 3 steps. Only include steps where a genuinely distinct "first repeat
after negative feedback" occurs. Do not over-annotate.

Respond with ONLY the step numbers as a comma-separated list (e.g. 10,40). No explanation, no JSON, no other text."""


# ---------------------------------------------------------------------------
# IoU and grouping logic (mirrors annotation_ui.py — kept here to avoid Flask)
# ---------------------------------------------------------------------------

def iou(a, b):
    a0, a1 = a
    b0, b1 = b
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


def build_episode_groups(ep: dict) -> list[dict]:
    """Mirror of annotation_ui.build_episode_groups (no Flask dependency)."""
    fa_list = ep.get("claude") or []
    fb_list = ep.get("gemini") or []
    items = []

    used_b = set()
    matched_pairs = []
    for fa in fa_list:
        wa = fa.get("where", [0, 0])
        bi, bs = -1, 0.0
        for j, fb in enumerate(fb_list):
            if j in used_b:
                continue
            s = iou(wa, fb.get("where", [0, 0]))
            if s > bs:
                bs, bi = s, j
        if bs >= IOU_THRESHOLD and bi >= 0:
            matched_pairs.append((fa, fb_list[bi], bs))
            used_b.add(bi)

    matched_a_ids = {id(m[0]) for m in matched_pairs}
    unm_a = [fa for fa in fa_list if id(fa) not in matched_a_ids]
    unm_b = [fb for j, fb in enumerate(fb_list) if j not in used_b]

    for fa, fb, score in matched_pairs:
        if fa.get("tier") == "core" or fb.get("tier") == "core":
            items.append({"group": "matched", "iou": round(score, 3), "claude": fa, "gemini": fb})

    used_low_b = set()
    for fa in unm_a:
        for j, fb in enumerate(unm_b):
            if j in used_low_b:
                continue
            s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
            if 0 < s < IOU_THRESHOLD:
                if fa.get("tier") == "core" or fb.get("tier") == "core":
                    items.append({"group": "low_conf", "iou": round(s, 3), "claude": fa, "gemini": fb})
                    used_low_b.add(j)
                    break

    low_used_a = {id(item["claude"]) for item in items if item["group"] == "low_conf" and item.get("claude")}
    for fa in unm_a:
        if id(fa) not in low_used_a and fa.get("tier") == "core":
            items.append({"group": "claude_only", "iou": None, "claude": fa, "gemini": None})

    low_used_b_ids = {id(item["gemini"]) for item in items if item["group"] == "low_conf" and item.get("gemini")}
    for fb in unm_b:
        if id(fb) not in low_used_b_ids and fb.get("tier") == "core":
            items.append({"group": "gemini_only", "iou": None, "claude": None, "gemini": fb})

    # Fallback: show all when no core failures found
    if not items:
        for fa, fb, score in matched_pairs:
            items.append({"group": "matched", "iou": round(score, 3), "claude": fa, "gemini": fb, "fallback": True})
        used_low_b2 = set()
        for fa in unm_a:
            for j, fb in enumerate(unm_b):
                if j in used_low_b2:
                    continue
                s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
                if 0 < s < IOU_THRESHOLD:
                    items.append({"group": "low_conf", "iou": round(s, 3), "claude": fa, "gemini": fb, "fallback": True})
                    used_low_b2.add(j)
                    break
        low_used_a2 = {id(item["claude"]) for item in items if item.get("fallback") and item.get("claude")}
        for fa in unm_a:
            if id(fa) not in low_used_a2:
                items.append({"group": "claude_only", "iou": None, "claude": fa, "gemini": None, "fallback": True})
        low_used_b2_ids = {id(item["gemini"]) for item in items if item.get("fallback") and item.get("gemini")}
        for fb in unm_b:
            if id(fb) not in low_used_b2_ids:
                items.append({"group": "gemini_only", "iou": None, "claude": None, "gemini": fb, "fallback": True})

    return items


# ---------------------------------------------------------------------------
# Human annotation filter
# ---------------------------------------------------------------------------

def get_kept_failure_ids(game: str, ai_snaps: dict) -> set[tuple[str, str, str]] | None:
    """
    Returns set of (ep_id, annotator, failure_id) for all human-kept items.
    Returns None if no human annotation file exists for this game.
    """
    human_path = OUTPUT / "human" / f"{game}_human.json"
    if not human_path.exists():
        return None

    human_data = json.loads(human_path.read_text())
    kept: set[tuple[str, str, str]] = set()

    for ep_id, ep_human in human_data["episodes"].items():
        ep_ai = ai_snaps.get(ep_id)
        if not ep_ai:
            continue
        groups = build_episode_groups(ep_ai)
        items = ep_human.get("items", {})
        for idx_str, item in items.items():
            if not item.get("keep"):
                continue
            idx = int(idx_str)
            if idx >= len(groups):
                continue
            group = groups[idx]
            choice = item.get("diagnosis_choice")
            # Determine which annotators to reannotate based on choice
            if choice == "A":
                annotators = ["claude"]
            elif choice == "B":
                annotators = ["gemini"]
            elif choice in ("C", "D", None):
                annotators = ["claude", "gemini"]
            else:
                annotators = ["claude", "gemini"]
            for annotator in annotators:
                f = group.get(annotator)
                if f and f.get("failure_id"):
                    kept.add((ep_id, annotator, f["failure_id"]))

    return kept


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def build_trajectory_window(
    trajectory: list[dict], where: list[int], context: int = 5, full: bool = False
) -> str:
    """Return a text representation of trajectory steps around the where range."""
    first, last = where
    if full:
        lo, hi = 0, len(trajectory) - 1
    else:
        lo = max(0, first - context)
        hi = min(len(trajectory) - 1, last + context)

    lines = []
    for step in trajectory:
        s = step["step"]
        if s < lo or s > hi:
            continue
        marker = ">>>" if first <= s <= last else "   "
        obs = step.get("obs", "").strip().replace("\n", " ")[:200]
        action = step.get("action", "").strip()
        lines.append(f"{marker} [Step {s}] OBS: {obs}")
        if action:
            lines.append(f"         ACTION: {action}")
    return "\n".join(lines)


def load_trajectory(game: str, ep_id: str) -> list[dict]:
    """Load trajectory from the episode log file."""
    m = re.search(r"_(gpt41|qwen332b)_(\d+)$", ep_id)
    if not m:
        return []
    model_tag = m.group(1)
    ep_num    = int(m.group(2))

    suffix = "" if model_tag == "gpt41" else "_qwen3"
    ann_path = OUTPUT / f"{game}_annotations{suffix}.json"
    if not ann_path.exists():
        return []
    ann_data = json.loads(ann_path.read_text())
    run_dir  = Path(ann_data["run_dir"])

    pattern = f"episode_{ep_num:03d}_*.txt"
    matches = list(run_dir.glob(pattern))
    if not matches:
        return []

    log_path = matches[0]
    steps = []
    current: dict = {}
    section = None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("[STEP]"):
            if current:
                steps.append(current)
            current = {"step": int(line.split()[1]), "obs": "", "action": ""}
            section = None
        elif line.startswith("[OBS]"):
            section = "obs"
            current["obs"] = line[5:].strip()
        elif line.startswith("[INV]"):
            section = None
        elif line.startswith("[RAW_LLM_OUTPUT]"):
            section = None
        elif line.startswith("ACTION:"):
            current["action"] = line[7:].strip()
            section = None
        elif line.startswith("---") or line.startswith("==="):
            section = None
        elif section == "obs":
            current["obs"] += "\n" + line
    if current:
        steps.append(current)
    return steps


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def ask_claude(client: Anthropic, failure: dict, trajectory: list[dict]) -> list[int] | None:
    """
    Ask Claude to identify step(s) for an operation failure.
    feedback_blindness: returns 1–3 steps. Others: returns 1 step.
    """
    ftype = failure.get("type", "")
    rule  = STEP_DEFINITIONS.get(ftype, "the most relevant single step")
    where = failure.get("where", [0, 0])
    why   = failure.get("why", "")

    is_fb  = (ftype == "operation/feedback_blindness")
    window = build_trajectory_window(trajectory, where, context=5, full=is_fb)

    if is_fb:
        system   = SYSTEM_PROMPT_MULTI
        question = "List each distinct step (at most 3) where a new feedback_blindness occurrence begins."
    else:
        system   = SYSTEM_PROMPT_SINGLE
        question = "What is the single step number where this failure occurs?"

    user_msg = f"""Failure type: {ftype}
Localization rule: {rule}
Failure description: {why}
Current annotated range: steps {where[0]}–{where[1]}

Trajectory window (steps marked >>> are within the annotated range):
{window}

{question}"""

    raw = ""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=32,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        if is_fb:
            steps = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            if not steps:
                raise ValueError(f"no integers found in: {raw!r}")
            return steps[:3]
        else:
            return [int(raw)]
    except Exception as e:
        print(f"    ERROR parsing response: {e} | raw: {raw!r}")
        return None


# ---------------------------------------------------------------------------
# Per-game processing
# ---------------------------------------------------------------------------

def process_game(game: str, client: Anthropic, dry_run: bool = False):
    path = OUTPUT / f"{game}_ai_annotations.json"
    data = json.loads(path.read_text())
    snapshots = data["snapshots"]
    ai_snaps  = {ep["id"]: ep for ep in snapshots}

    # Load human-kept failure ids if available; otherwise process all
    kept_ids = get_kept_failure_ids(game, ai_snaps)
    if kept_ids is not None:
        print(f"  [{game}] human annotations found — processing {len(kept_ids)} kept failure(s)")
    else:
        print(f"  [{game}] no human annotations — processing all operation failures")

    n_updated = 0
    n_skipped = 0
    n_failed  = 0

    for ep in snapshots:
        ep_id      = ep["id"]
        trajectory = load_trajectory(game, ep_id)
        if not trajectory:
            print(f"  [{ep_id}] no trajectory found, skipping")
            continue

        changed = False
        for annotator in ["claude", "gemini"]:
            failures     = ep.get(annotator) or []
            new_failures = []
            for failure in failures:
                ftype = failure.get("type", "")
                if ftype not in OPERATION_TYPES:
                    new_failures.append(failure)
                    continue

                old_where = failure.get("where", [0, 0])
                if old_where[0] == old_where[1]:
                    new_failures.append(failure)
                    continue

                # Filter by human-kept items when available
                if kept_ids is not None:
                    fid = failure.get("failure_id", "")
                    if (ep_id, annotator, fid) not in kept_ids:
                        new_failures.append(failure)
                        n_skipped += 1
                        continue

                print(f"  [{ep_id}] {annotator} {ftype} {old_where} → querying Claude...")
                steps = ask_claude(client, failure, trajectory)

                if steps is None:
                    print(f"    FAILED, keeping original range")
                    new_failures.append(failure)
                    n_failed += 1
                    continue

                print(f"    → steps {steps}")
                if not dry_run:
                    failure["where"] = [steps[0], steps[0]]
                    new_failures.append(failure)
                    for extra_step in steps[1:]:
                        extra = dict(failure)
                        extra["where"] = [extra_step, extra_step]
                        extra["failure_id"] = failure.get("failure_id", "") + f"_r{extra_step}"
                        new_failures.append(extra)
                else:
                    new_failures.append(failure)
                n_updated += len(steps)
                changed = True

                time.sleep(0.3)

            if not dry_run and changed:
                ep[annotator] = new_failures

    print(f"  {game}: {n_updated} updated, {n_skipped} skipped (not human-kept), {n_failed} failed")
    if not dry_run and n_updated > 0:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  → saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--game",  choices=GAMES)
    group.add_argument("--games", nargs="+", choices=GAMES)
    group.add_argument("--all",   action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without saving")
    args = parser.parse_args()

    if args.all:
        games = GAMES
    elif args.games:
        games = args.games
    elif args.game:
        games = [args.game]
    else:
        parser.print_help()
        return

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if len(games) == 1:
        print(f"\n=== {games[0]} ===")
        process_game(games[0], client, dry_run=args.dry_run)
    else:
        # Run multiple games in parallel threads
        with ThreadPoolExecutor(max_workers=len(games)) as executor:
            futures = {
                executor.submit(process_game, g, client, args.dry_run): g
                for g in games
            }
            for future in as_completed(futures):
                game = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n=== {game} CRASHED: {e} ===")


if __name__ == "__main__":
    main()
