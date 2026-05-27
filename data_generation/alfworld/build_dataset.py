"""
build_dataset.py  —  Build the ROGUE snapshot dataset from AlfWorld AI annotations.

Each snapshot file becomes a separate dataset entry with a unique ID:
  <original_id>_<model_clean>_inst<NNN>
e.g. alfworld_pick_clean_evomemory_01_Qwen_Qwen3-32B_inst004

Adjudication rules (human-in-the-loop via script):
  - matched (IoU >= 0.3), both core      → keep Claude's diagnosis; flag cross-category
  - matched (IoU >= 0.3), one core       → keep the core annotator's diagnosis
  - matched (IoU >= 0.3), both marginal  → skip (goes to marginal_failure only)
  - low_conf (0 < IoU < 0.3), at least one core → keep each core separately
  - claude_only core                     → keep Claude's diagnosis
  - gemini_only core                     → keep Gemini's diagnosis
  - fallback (all marginal after filter) → apply same rules without tier filter

Usage:
    conda run -n Fraud python build_dataset.py [--output_dir ../../final_dataset/alfworld]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
SNAPSHOTS_DIR = BASE.parent / "output"
ANN_FILE = BASE / "output" / "alfworld_all_ai_annotations.json"

IOU_THRESHOLD = 0.3


# ── Helpers ───────────────────────────────────────────────────────────────────

def iou(a, b):
    a0, a1 = a; b0, b1 = b
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


def category(t: str) -> str:
    return t.split("/")[0] if t else ""


def make_unique_id(snap: dict, path: Path) -> str:
    """<original_id>_<model_clean>_inst<NNN> from snapshot fields + file path."""
    original_id = snap.get("id", "unknown")
    model_clean = snap.get("model", "").replace("/", "_")
    # filename: "instance_004_ep_01" → inst004
    parts = path.stem.split("_")
    inst_num = parts[1] if len(parts) >= 2 else "000"
    return f"{original_id}_{model_clean}_inst{inst_num}"


def load_failed_snapshots() -> list[tuple[Path, dict]]:
    """All won=False snapshots in sorted path order — must match ai_annotate.py ordering."""
    result = []
    for path in sorted(SNAPSHOTS_DIR.rglob("*.json")):
        if path.name == "summary.json":
            continue
        snap = json.loads(path.read_text(encoding="utf-8"))
        if snap.get("won", True):
            continue
        result.append((path, snap))
    return result


def load_won_snapshots() -> list[tuple[Path, dict]]:
    """All won=True snapshots, deduped to latest run per unique_id."""
    seen: dict[str, tuple[Path, dict]] = {}
    for path in sorted(SNAPSHOTS_DIR.rglob("*.json")):
        if path.name == "summary.json":
            continue
        snap = json.loads(path.read_text(encoding="utf-8"))
        if snap.get("won", False):
            uid = make_unique_id(snap, path)
            seen[uid] = (path, snap)   # later (newer) paths overwrite earlier ones
    return list(seen.values())


def strip_trajectory(traj: list) -> list:
    out = []
    for step in traj:
        entry = {"step": step["step"], "obs": step.get("obs", "")}
        if step.get("action") is not None:
            entry["action"] = step["action"]
        if "progress" in step:
            entry["progress"] = step["progress"]
        out.append(entry)
    return out


# ── Item grouping (mirrors JTTL annotation_ui logic) ──────────────────────────

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

    low_used_a = {id(item["claude"]) for item in items if item.get("claude")}
    for fa in unm_a:
        if id(fa) not in low_used_a and fa.get("tier") == "core":
            items.append({"group": "claude_only", "iou": None, "claude": fa, "gemini": None})

    low_used_b_ids = {id(item["gemini"]) for item in items if item.get("gemini")}
    for fb in unm_b:
        if id(fb) not in low_used_b_ids and fb.get("tier") == "core":
            items.append({"group": "gemini_only", "iou": None, "claude": None, "gemini": fb})

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


# ── Adjudication ──────────────────────────────────────────────────────────────

def adjudicate(ep: dict, original_id: str, human_decisions: dict, unique_id: str = "") -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (core_failures, marginal_failures, uncertain_items).
    Looks up human_decisions first by unique_id (instance-specific), then by original_id (fallback).
    """
    items = build_items(ep)
    ep_overrides = human_decisions.get(unique_id) or human_decisions.get(original_id, {})
    core_failures = []
    core_wheres = []
    uncertain = []

    for i, item in enumerate(items):
        c = item.get("claude")
        g = item.get("gemini")
        group = item["group"]
        override = ep_overrides.get(str(i))

        if override:
            decision = override.get("decision")
            if decision == "drop":
                continue
            if decision == "A" and c:
                entry = {"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")}
                if not override.get("force_marginal"):
                    core_failures.append(entry)
                    core_wheres.append(entry["where"])
                continue
            if decision == "B" and g:
                entry = {"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")}
                if not override.get("force_marginal"):
                    core_failures.append(entry)
                    core_wheres.append(entry["where"])
                continue
            if decision == "D":
                entry = {
                    "type":      override.get("custom_type", "unknown"),
                    "where":     override.get("custom_where", [0, 0]),
                    "diagnosis": override.get("custom_diagnosis", ""),
                }
                if not override.get("force_marginal"):
                    core_failures.append(entry)
                    core_wheres.append(entry["where"])
                continue

        if group == "matched":
            c_tier = c.get("tier") if c else None
            g_tier = g.get("tier") if g else None
            both_core = c_tier == "core" and g_tier == "core"
            one_core  = (c_tier == "core") != (g_tier == "core")

            if both_core:
                if category(c.get("type", "")) != category(g.get("type", "")):
                    uncertain.append({
                        "ep_id": original_id, "group": group, "iou": item.get("iou"),
                        "reason": "matched both-core but cross-category type disagreement",
                        "auto_decision": "keep Claude's diagnosis",
                        "claude": {"type": c.get("type"), "where": c.get("where"), "why": c.get("why")},
                        "gemini": {"type": g.get("type"), "where": g.get("where"), "why": g.get("why")},
                    })
                core_failures.append({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                core_wheres.append(c.get("where", [0, 0]))
            elif one_core:
                chosen = c if c_tier == "core" else g
                core_failures.append({"type": chosen.get("type", "unknown"), "where": chosen.get("where", [0, 0]), "diagnosis": chosen.get("why", "")})
                core_wheres.append(chosen.get("where", [0, 0]))

        elif group == "low_conf":
            c_tier = c.get("tier") if c else None
            g_tier = g.get("tier") if g else None
            both_core = c_tier == "core" and g_tier == "core"

            if both_core:
                if category(c.get("type", "")) != category(g.get("type", "")):
                    uncertain.append({
                        "ep_id": original_id, "group": group, "iou": item.get("iou"),
                        "reason": "low-conf both-core cross-category: possibly two distinct failures",
                        "auto_decision": "keep both as separate core failures",
                        "claude": {"type": c.get("type"), "where": c.get("where"), "why": c.get("why")},
                        "gemini": {"type": g.get("type"), "where": g.get("where"), "why": g.get("why")},
                    })
                core_failures.append({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                core_wheres.append(c.get("where", [0, 0]))
                core_failures.append({"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")})
                core_wheres.append(g.get("where", [0, 0]))
            elif c_tier == "core":
                core_failures.append({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                core_wheres.append(c.get("where", [0, 0]))
            elif g_tier == "core":
                core_failures.append({"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")})
                core_wheres.append(g.get("where", [0, 0]))

        elif group == "claude_only":
            if c and c.get("tier") == "core":
                core_failures.append({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                core_wheres.append(c.get("where", [0, 0]))

        elif group == "gemini_only":
            if g and g.get("tier") == "core":
                core_failures.append({"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")})
                core_wheres.append(g.get("where", [0, 0]))

    core_failures.sort(key=lambda f: f["where"][0])

    fa_list = ep.get("claude") or []
    fb_list = ep.get("gemini") or []
    all_ai = []
    used_b = set()
    for fa in fa_list:
        best_iou, best_j = 0.0, -1
        for j, fb in enumerate(fb_list):
            if j in used_b:
                continue
            s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
            if s > best_iou:
                best_iou, best_j = s, j
        all_ai.append({"type": fa.get("type", "unknown"), "where": fa.get("where", [0, 0])})
        if best_iou >= IOU_THRESHOLD and best_j >= 0:
            used_b.add(best_j)
    for j, fb in enumerate(fb_list):
        if j not in used_b:
            all_ai.append({"type": fb.get("type", "unknown"), "where": fb.get("where", [0, 0])})

    marginal_failures = [
        f for f in all_ai
        if not any(iou(f["where"], cw) >= IOU_THRESHOLD for cw in core_wheres)
    ]
    marginal_failures.sort(key=lambda f: f["where"][0])

    return core_failures, marginal_failures, uncertain


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="../../final_dataset/alfworld")
    parser.add_argument("--decisions", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Loading snapshots...")
    failed_snaps = load_failed_snapshots()
    won_snaps    = load_won_snapshots()
    print(f"  {len(failed_snaps)} failed, {len(won_snaps)} won snapshot files")

    print("Loading AI annotations...")
    ann_data = json.loads(ANN_FILE.read_text(encoding="utf-8"))
    ann_entries = ann_data["snapshots"]
    assert len(ann_entries) == len(failed_snaps), (
        f"Annotation count ({len(ann_entries)}) != failed snapshot count ({len(failed_snaps)})"
    )
    print(f"  {len(ann_entries)} annotation entries — positional match verified")

    human_decisions = {}
    decisions_path = Path(args.decisions) if args.decisions else BASE / "human_decisions.json"
    if decisions_path.exists():
        raw = json.loads(decisions_path.read_text(encoding="utf-8"))
        human_decisions = raw.get("decisions", {})
        print(f"  {len(human_decisions)} episode overrides loaded from {decisions_path.name}")

    # unique_id → episode_out; later entries (newer runs) overwrite earlier ones
    failed_by_uid: dict[str, dict] = {}
    all_uncertain = []

    # ── Failed episodes ───────────────────────────────────────────────────────
    for (path, snap), ann in zip(failed_snaps, ann_entries):
        original_id = snap.get("id", "unknown")
        unique_id   = make_unique_id(snap, path)

        core_failures, marginal_failures, uncertain = adjudicate(ann, original_id, human_decisions, unique_id)
        all_uncertain.extend(uncertain)

        task_short = snap.get("task_short", "unknown")
        traj       = strip_trajectory(snap.get("trajectory", []))

        failed_by_uid[unique_id] = {
            "id":        unique_id,
            "game":      task_short,
            "task_type": snap.get("task_type", ""),
            "model":     snap.get("model", ""),
            "won":       False,
            "snapshot": {
                "final_score": snap.get("progress", 0.0),
                "max_score":   1.0,
                "n_steps":     snap.get("n_steps") or len(traj),
                "trajectory":  traj,
            },
            "failure_instances": {
                "core_failure":     core_failures,
                "marginal_failure": marginal_failures,
            },
        }

    # Warn on no-core after dedup (only count unique IDs)
    n_no_core = sum(
        1 for ep in failed_by_uid.values()
        if not ep["failure_instances"]["core_failure"]
    )
    if n_no_core:
        for ep in failed_by_uid.values():
            if not ep["failure_instances"]["core_failure"]:
                print(f"  WARNING: {ep['id']} has no core failures after adjudication!")

    by_task: dict[str, list] = defaultdict(list)
    for ep in failed_by_uid.values():
        by_task[ep["game"]].append(ep)

    # ── Won episodes ──────────────────────────────────────────────────────────
    for path, snap in won_snaps:
        unique_id  = make_unique_id(snap, path)
        if unique_id in failed_by_uid:
            continue   # same instance already represented as a failed episode
        task_short = snap.get("task_short", "unknown")
        traj       = strip_trajectory(snap.get("trajectory", []))

        by_task[task_short].append({
            "id":        unique_id,
            "game":      task_short,
            "task_type": snap.get("task_type", ""),
            "model":     snap.get("model", ""),
            "won":       True,
            "snapshot": {
                "final_score": snap.get("progress", 1.0),
                "max_score":   1.0,
                "n_steps":     snap.get("n_steps") or len(traj),
                "trajectory":  traj,
            },
            "failure_instances": {
                "core_failure":     [],
                "marginal_failure": [],
            },
        })

    # ── Write outputs ─────────────────────────────────────────────────────────
    all_episodes = []
    for task_short, episodes in sorted(by_task.items()):
        task_dir = output_dir / task_short
        task_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "version":    "1.0",
            "game":       task_short,
            "created_at": datetime.now().isoformat(),
            "n_episodes": len(episodes),
            "episodes":   episodes,
        }
        out_path = task_dir / "snapshots.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        won_n   = sum(1 for e in episodes if e["won"])
        failed_n = len(episodes) - won_n
        print(f"  {task_short}: {len(episodes)} episodes (failed={failed_n}, won={won_n}) → {out_path}")
        all_episodes.extend(episodes)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "version":    "1.0",
        "created_at": datetime.now().isoformat(),
        "n_episodes": len(all_episodes),
        "episodes":   all_episodes,
    }
    (output_dir / "all.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    won_total   = sum(1 for e in all_episodes if e["won"])
    failed_total = len(all_episodes) - won_total
    print(f"\nCombined: {len(all_episodes)} episodes (failed={failed_total}, won={won_total}) → {output_dir / 'all.json'}")

    uncertain_out = {
        "created_at": datetime.now().isoformat(),
        "n_items":    len(all_uncertain),
        "note": "Items flagged for human review. Override via human_decisions.json keyed by original episode ID.",
        "items": all_uncertain,
    }
    (output_dir / "uncertain_cases.json").write_text(json.dumps(uncertain_out, indent=2, ensure_ascii=False))
    print(f"Uncertain cases: {len(all_uncertain)} items → {output_dir / 'uncertain_cases.json'}")

    if n_no_core > 0:
        print(f"\nWARNING: {n_no_core} episodes have no core failures after adjudication.")


if __name__ == "__main__":
    main()
