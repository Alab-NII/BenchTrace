"""
build_dataset.py  —  Build the ROGUE snapshot dataset from human annotations.

For each annotated episode, produces:
  { id, game, model, snapshot: {trajectory, final_score, max_score, n_steps},
    reflection_golden: {n_failures, failures: [{type, where, diagnosis}]} }

Usage:
    conda run -n Fraud python build_dataset.py --game detective
    conda run -n Fraud python build_dataset.py --all
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
OUTPUT = BASE / "output"
HUMAN_DIR = OUTPUT / "human"
DATASET_DIR = OUTPUT / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

GAMES = ["detective", "library", "zork1", "zork3", "balances", "temple"]

MODEL_ANN_SUFFIX = {
    "gpt41":    "",
    "qwen332b": "_qwen3",
}
MODEL_NAME = {
    "gpt41":    "gpt-4.1",
    "qwen332b": "qwen3-32b",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_run_dir(game: str, model_tag: str) -> Path | None:
    suffix = MODEL_ANN_SUFFIX.get(model_tag, "")
    path = OUTPUT / f"{game}_annotations{suffix}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Path(data["run_dir"])


def get_episode_meta(game: str, model_tag: str, ep_num: int) -> dict:
    """Return final_score, max_score, n_steps from annotations file."""
    suffix = MODEL_ANN_SUFFIX.get(model_tag, "")
    path = OUTPUT / f"{game}_annotations{suffix}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    ep = next((e for e in data["episodes"] if e.get("episode") == ep_num), None)
    if ep is None:
        return {"max_score": data.get("max_score")}
    return {
        "final_score": ep.get("final_score"),
        "max_score":   ep.get("max_score") or data.get("max_score"),
        "n_steps":     ep.get("n_steps"),
    }


def find_log_file(run_dir: Path, ep_num: int) -> Path | None:
    matches = list(run_dir.glob(f"episode_{ep_num:03d}_*.txt"))
    return matches[0] if matches else None


FORMAT_ERROR_PATTERNS = [
    "i only understood you as far as wanting to",
    "i don't understand that.",
    "please give one of the answers above",
]

def detect_format_failures(trajectory: list[dict]) -> list[dict]:
    """
    Detect steps where Jericho received a malformed action.
    Returns list of {type, where, tier, diagnosis} entries.
    First occurrence = core, rest = marginal.
    """
    bad_steps = []
    for step in trajectory:
        obs = step.get("obs", "").lower()
        if any(p in obs for p in FORMAT_ERROR_PATTERNS):
            bad_steps.append(step["step"])

    failures = []
    for i, s in enumerate(bad_steps):
        tier = "core" if i == 0 else "marginal"
        failures.append({
            "type":      "system/format",
            "where":     [s, s],
            "tier":      tier,
            "diagnosis": "The agent output a malformed action; Jericho only partially understood it.",
        })
    return failures


def parse_trajectory(log_path: Path) -> list[dict]:
    steps = []
    current: dict = {}
    section = None

    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("[STEP]"):
            if current:
                steps.append(current)
            current = {"step": int(line.split()[1]), "obs": "", "action": "", "inv": ""}
            section = None
        elif line.startswith("[OBS]"):
            section = "obs"
            current["obs"] = line[5:].strip()
        elif line.startswith("[INV]"):
            section = "inv"
            current["inv"] = line[5:].strip()
        elif line.startswith("[RAW_LLM_OUTPUT]"):
            section = None
        elif line.startswith("ACTION:"):
            current["action"] = line[7:].strip()
            section = None
        elif line.startswith("----------") or line.startswith("=========="):
            section = None
        elif section == "obs":
            current["obs"] += "\n" + line
        elif section == "inv":
            current["inv"] += "\n" + line

    if current:
        steps.append(current)

    # Strip empty trailing fields
    for s in steps:
        s["obs"] = s["obs"].strip()
        s["inv"] = s["inv"].strip()
        if not s["inv"]:
            del s["inv"]

    return steps


def ann_to_core(ann: dict, custom_diagnosis: str | None = None) -> dict:
    """Convert one annotator's failure annotation to a core_failure entry."""
    return {
        "type":      ann.get("type", "unknown"),
        "where":     ann.get("where", [0, 0]),
        "diagnosis": custom_diagnosis if custom_diagnosis else ann.get("why", ""),
    }


def resolve_core_failures(item_human: dict, claude_ann: dict | None, gemini_ann: dict | None) -> list[dict]:
    """
    Return 0–2 core_failure entries based on the human's diagnosis choice:
      A → Claude's entry only
      B → Gemini's entry only
      C → both Claude and Gemini as separate entries
      D → custom diagnosis text, type/where from Claude (fallback: Gemini)
      None / no keep → empty list
    """
    choice = item_human.get("diagnosis_choice")
    if choice is None:
        return []
    if choice == "A":
        return [ann_to_core(claude_ann)] if claude_ann else []
    if choice == "B":
        return [ann_to_core(gemini_ann)] if gemini_ann else []
    if choice == "C":
        entries = []
        if claude_ann:
            entries.append(ann_to_core(claude_ann))
        if gemini_ann:
            entries.append(ann_to_core(gemini_ann))
        return entries
    if choice == "D":
        custom = item_human.get("diagnosis_custom", "").strip() or None
        ann = claude_ann or gemini_ann
        return [ann_to_core(ann, custom_diagnosis=custom)] if ann else []
    return []


# ── Core: build items for one episode (same logic as annotation_ui) ──────────

IOU_THRESHOLD = 0.3


def iou(a, b):
    a0, a1 = a; b0, b1 = b
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


def build_items(ep: dict) -> list[dict]:
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
            items.append({"claude": fa, "gemini": fb})

    used_low_b = set()
    for i, fa in enumerate(unm_a):
        for j, fb in enumerate(unm_b):
            if j in used_low_b:
                continue
            s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
            if 0 < s < IOU_THRESHOLD:
                if fa.get("tier") == "core" or fb.get("tier") == "core":
                    items.append({"claude": fa, "gemini": fb})
                    used_low_b.add(j)
                    break

    low_used_a = {id(item["claude"]) for item in items if item.get("claude")}
    for fa in unm_a:
        if id(fa) not in low_used_a and fa.get("tier") == "core":
            items.append({"claude": fa, "gemini": None})

    low_used_b_ids = {id(item["gemini"]) for item in items if item.get("gemini")}
    for fb in unm_b:
        if id(fb) not in low_used_b_ids and fb.get("tier") == "core":
            items.append({"claude": None, "gemini": fb})

    # Fallback: if nothing after core filter, show all
    if not items:
        for fa, fb, score in matched_pairs:
            items.append({"claude": fa, "gemini": fb})
        used_low_b2 = set()
        for i, fa in enumerate(unm_a):
            for j, fb in enumerate(unm_b):
                if j in used_low_b2:
                    continue
                s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
                if 0 < s < IOU_THRESHOLD:
                    items.append({"claude": fa, "gemini": fb})
                    used_low_b2.add(j)
                    break
        low_used_a2 = {id(item["claude"]) for item in items if item.get("claude")}
        for fa in unm_a:
            if id(fa) not in low_used_a2:
                items.append({"claude": fa, "gemini": None})
        low_used_b2_ids = {id(item["gemini"]) for item in items if item.get("gemini")}
        for fb in unm_b:
            if id(fb) not in low_used_b2_ids:
                items.append({"claude": None, "gemini": fb})

    return items


# ── Main builder ──────────────────────────────────────────────────────────────

def build_game(game: str) -> dict:
    ai_data  = json.loads((OUTPUT / f"{game}_ai_annotations.json").read_text())
    human_data = json.loads((HUMAN_DIR / f"{game}_human.json").read_text())

    episodes_out = []
    n_skipped = 0

    for ep in ai_data["snapshots"]:
        ep_id = ep["id"]
        ep_human = human_data["episodes"].get(ep_id)

        if ep_human is None or not ep_human.get("completed"):
            n_skipped += 1
            continue

        # Parse episode ID for model tag and number
        m = re.search(r"_(gpt41|qwen332b)_(\d+)$", ep_id)
        if not m:
            print(f"  WARNING: cannot parse model tag from {ep_id}, skipping")
            n_skipped += 1
            continue
        model_tag = m.group(1)
        ep_num = int(m.group(2))

        # Load trajectory
        run_dir = get_run_dir(game, model_tag)
        trajectory = []
        if run_dir:
            log_file = find_log_file(run_dir, ep_num)
            if log_file:
                trajectory = parse_trajectory(log_file)

        # Episode metadata
        meta = get_episode_meta(game, model_tag, ep_num)

        # Build failure_instances from human decisions
        items = build_items(ep)
        core_failures = []
        core_wheres = []  # track confirmed core step ranges for marginal dedup

        for i, item in enumerate(items):
            item_human = ep_human.get("items", {}).get(str(i), {})
            if item_human.get("keep") is not True:
                continue
            entries = resolve_core_failures(item_human, item.get("claude"), item.get("gemini"))
            if not entries:
                print(f"  WARNING: {ep_id} item {i} has keep=True but no diagnosis — skipping")
                continue
            for entry in entries:
                core_failures.append(entry)
                core_wheres.append(entry["where"])

        # Inject rule-based system/format failures from trajectory
        if trajectory:
            fmt_failures = detect_format_failures(trajectory)
            for ff in fmt_failures:
                if ff["tier"] == "core":
                    core_failures.append({"type": ff["type"], "where": ff["where"], "diagnosis": ff["diagnosis"]})
                    core_wheres.append(ff["where"])
                else:
                    # marginal format failures added later
                    pass
            fmt_marginal = [{"type": ff["type"], "where": ff["where"]} for ff in fmt_failures if ff["tier"] == "marginal"]
        else:
            fmt_marginal = []

        core_failures.sort(key=lambda f: f["where"][0])

        # Build marginal_failure: all AI-identified failures not confirmed as core
        # Collect all unique failures from Claude + Gemini, deduplicate by IoU
        fa_list = ep.get("claude") or []
        fb_list = ep.get("gemini") or []
        all_failures: list[dict] = []
        used_b = set()
        for fa in fa_list:
            best_iou, best_j = 0.0, -1
            for j, fb in enumerate(fb_list):
                if j in used_b:
                    continue
                s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
                if s > best_iou:
                    best_iou, best_j = s, j
            if best_iou >= IOU_THRESHOLD and best_j >= 0:
                all_failures.append({"type": fa.get("type", "unknown"), "where": fa.get("where", [0, 0])})
                used_b.add(best_j)
            else:
                all_failures.append({"type": fa.get("type", "unknown"), "where": fa.get("where", [0, 0])})
        for j, fb in enumerate(fb_list):
            if j not in used_b:
                all_failures.append({"type": fb.get("type", "unknown"), "where": fb.get("where", [0, 0])})

        # Exclude those already in core (IoU >= threshold with any confirmed core)
        marginal_failures = []
        for f in all_failures:
            is_core = any(iou(f["where"], cw) >= IOU_THRESHOLD for cw in core_wheres)
            if not is_core:
                marginal_failures.append(f)
        # Add rule-based marginal format failures (not already in core)
        for ff in fmt_marginal:
            is_core = any(iou(ff["where"], cw) >= IOU_THRESHOLD for cw in core_wheres)
            if not is_core:
                marginal_failures.append(ff)

        marginal_failures.sort(key=lambda f: f["where"][0])

        episodes_out.append({
            "id":    ep_id,
            "game":  game,
            "model": MODEL_NAME.get(model_tag, model_tag),
            "snapshot": {
                "final_score": meta.get("final_score"),
                "max_score":   meta.get("max_score"),
                "n_steps":     meta.get("n_steps") or len(trajectory),
                "trajectory":  trajectory,
            },
            "failure_instances": {
                "core_failure":     core_failures,
                "marginal_failure": marginal_failures,
            },
        })

    print(f"  {game}: {len(episodes_out)} episodes built, {n_skipped} skipped (not completed)")
    return {
        "version":    "1.0",
        "game":       game,
        "created_at": datetime.now().isoformat(),
        "n_episodes": len(episodes_out),
        "episodes":   episodes_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=GAMES, help="Build dataset for one game")
    parser.add_argument("--all", action="store_true", help="Build all games")
    args = parser.parse_args()

    games_to_build = GAMES if args.all else ([args.game] if args.game else None)
    if not games_to_build:
        parser.print_help()
        return

    all_episodes = []
    for game in games_to_build:
        human_path = HUMAN_DIR / f"{game}_human.json"
        if not human_path.exists():
            print(f"  {game}: no human annotations yet, skipping")
            continue
        print(f"Building {game}...")
        result = build_game(game)
        out_path = DATASET_DIR / f"{game}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  → saved to {out_path}")
        all_episodes.extend(result["episodes"])

    if len(games_to_build) > 1 and all_episodes:
        combined = {
            "version":    "1.0",
            "created_at": datetime.now().isoformat(),
            "n_episodes": len(all_episodes),
            "episodes":   all_episodes,
        }
        out_path = DATASET_DIR / "all.json"
        out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
        print(f"\nCombined dataset: {len(all_episodes)} episodes → {out_path}")


if __name__ == "__main__":
    main()
