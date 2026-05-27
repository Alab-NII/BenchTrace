"""
build_dataset.py  —  Build the ROGUE snapshot dataset from GroupTravelPlanning AI annotations.

Each snapshot file becomes a separate dataset entry with a unique ID:
  <original_id>_<model_clean>
e.g. group_travel_task005_ep00_gpt-4.1

Adjudication rules (human-in-the-loop via human_decisions.json):
  - matched (IoU >= 0.3), both core      → keep Claude's diagnosis; flag cross-category
  - matched (IoU >= 0.3), one core       → keep the core annotator's diagnosis
  - matched (IoU >= 0.3), both marginal  → skip (goes to marginal_failure only)
  - low_conf (0 < IoU < 0.3), at least one core → keep each core separately; flag cross-category
  - claude_only core                     → keep Claude's diagnosis
  - gemini_only core                     → keep Gemini's diagnosis
  - fallback (all marginal after filter) → apply same rules without tier filter

Usage:
    conda run -n Fraud python build_dataset.py [--output_dir ../../final_dataset/group_travel_planning]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
SNAPSHOTS_DIR = BASE.parent / "output"
ANN_FILE = BASE / "output" / "group_travel_planning_all_ai_annotations.json"

IOU_THRESHOLD = 0.3


# ── Helpers ───────────────────────────────────────────────────────────────────

def norm_where(w):
    """Normalize where to [start, end]; single-element [n] → [n, n]."""
    if len(w) == 1:
        return [w[0], w[0]]
    return w


def norm_op_where(entry: dict) -> dict:
    """For operation/* types, narrow range [a,b] to [a,a] — operation failures are single-step."""
    t = entry.get("type", "")
    w = entry.get("where", [0, 0])
    if t.startswith("operation/") and len(w) == 2 and w[0] != w[1]:
        entry = dict(entry)
        entry["where"] = [w[0], w[0]]
    return entry


def iou(a, b):
    a, b = norm_where(a), norm_where(b)
    a0, a1 = a; b0, b1 = b
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


def category(t: str) -> str:
    return t.split("/")[0] if t else ""


def make_unique_id(snap: dict, path: Path) -> str:
    """<original_id>_<model_clean> — stable across timestamp directories."""
    original_id = snap.get("id", "unknown")
    model_clean = snap.get("model", "").replace("/", "_")
    return f"{original_id}_{model_clean}"


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
            seen[uid] = (path, snap)
    return list(seen.values())


def strip_trajectory(traj: list) -> list:
    out = []
    for step in traj:
        entry = {
            "step": step["step"],
            "subtask_idx": step.get("subtask_idx"),
            "obs": step.get("obs", ""),
        }
        if step.get("action") is not None:
            entry["action"] = step["action"]
        if "subtask_progress" in step:
            entry["subtask_progress"] = step["subtask_progress"]
        if "progress" in step:
            entry["progress"] = step["progress"]
        out.append(entry)
    return out


# ── Item grouping ─────────────────────────────────────────────────────────────

def build_items(ep: dict) -> list[dict]:
    fa_list = ep.get("claude") or []
    fb_list = ep.get("gemini") or []
    for item in fa_list + fb_list:
        if "where" in item:
            item["where"] = norm_where(item["where"])
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
                entry = norm_op_where({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                if not override.get("force_marginal"):
                    core_failures.append(entry)
                    core_wheres.append(entry["where"])
                continue
            if decision == "B" and g:
                entry = norm_op_where({"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")})
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
                entry = norm_op_where({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                core_failures.append(entry)
                core_wheres.append(entry["where"])
            elif one_core:
                chosen = c if c_tier == "core" else g
                entry = norm_op_where({"type": chosen.get("type", "unknown"), "where": chosen.get("where", [0, 0]), "diagnosis": chosen.get("why", "")})
                core_failures.append(entry)
                core_wheres.append(entry["where"])

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
                entry_c = norm_op_where({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                entry_g = norm_op_where({"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")})
                core_failures.append(entry_c)
                core_wheres.append(entry_c["where"])
                core_failures.append(entry_g)
                core_wheres.append(entry_g["where"])
            elif c_tier == "core":
                entry = norm_op_where({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                core_failures.append(entry)
                core_wheres.append(entry["where"])
            elif g_tier == "core":
                entry = norm_op_where({"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")})
                core_failures.append(entry)
                core_wheres.append(entry["where"])

        elif group == "claude_only":
            if c and c.get("tier") == "core":
                entry = norm_op_where({"type": c.get("type", "unknown"), "where": c.get("where", [0, 0]), "diagnosis": c.get("why", "")})
                core_failures.append(entry)
                core_wheres.append(entry["where"])

        elif group == "gemini_only":
            if g and g.get("tier") == "core":
                entry = norm_op_where({"type": g.get("type", "unknown"), "where": g.get("where", [0, 0]), "diagnosis": g.get("why", "")})
                core_failures.append(entry)
                core_wheres.append(entry["where"])

    if not core_failures:
        print(f"  WARNING: no core failures for {unique_id or original_id}")
        uncertain.append({
            "ep_id": original_id,
            "group": "no_core",
            "reason": "no core failure identified after adjudication",
            "auto_decision": "manual review required",
        })

    all_ai = []
    for fa in (ep.get("claude") or []):
        all_ai.append(norm_op_where({"type": fa.get("type", ""), "where": fa.get("where", [0, 0]), "diagnosis": fa.get("why", "")}))
    for fb in (ep.get("gemini") or []):
        all_ai.append(norm_op_where({"type": fb.get("type", ""), "where": fb.get("where", [0, 0]), "diagnosis": fb.get("why", "")}))

    marginal_failures = [
        norm_op_where(f) for f in all_ai
        if not any(iou(f["where"], cw) >= IOU_THRESHOLD for cw in core_wheres)
    ]

    return core_failures, marginal_failures, uncertain


# ── Dataset entry builder ─────────────────────────────────────────────────────

def make_entry(snap: dict, path: Path, ann_ep: dict | None, human_decisions: dict) -> dict:
    unique_id = make_unique_id(snap, path)
    original_id = snap.get("id", "unknown")
    traj = snap.get("trajectory", [])
    task_id = snap.get("task_id", "unknown")
    n = snap.get("n_subtasks") or len(snap.get("trajectory", [])) // 2
    game_label = f"{n}_travelers"

    entry = {
        "id": unique_id,
        "game": game_label,
        "task": snap.get("task", "group_travel_planning"),
        "task_id": task_id,
        "framework": snap.get("framework", ""),
        "model": snap.get("model", ""),
        "won": snap.get("won", False),
        "snapshot": {
            "final_score": snap.get("progress", 0.0),
            "max_score": 1.0,
            "n_steps": snap.get("n_subtasks") or len(traj),
            "trajectory": strip_trajectory(traj),
        },
        "failure_instances": {
            "core_failure": [],
            "marginal_failure": [],
        },
    }

    if not snap.get("won", False) and ann_ep is not None:
        core, marginal, _ = adjudicate(ann_ep, original_id, human_decisions, unique_id)
        entry["failure_instances"]["core_failure"] = core
        entry["failure_instances"]["marginal_failure"] = marginal

    return entry


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    output_dir = Path(args.output_dir)

    print("Loading snapshots...")
    failed_snaps = load_failed_snapshots()
    won_snaps = load_won_snapshots()
    print(f"  {len(failed_snaps)} failed, {len(won_snaps)} won snapshot files")

    print("Loading AI annotations...")
    with open(ANN_FILE, encoding="utf-8") as f:
        ann_data = json.load(f)
    ann_entries = ann_data["snapshots"]
    assert len(ann_entries) == len(failed_snaps), (
        f"Annotation count ({len(ann_entries)}) != failed snapshot count ({len(failed_snaps)})"
    )
    print(f"  {len(ann_entries)} annotation entries — positional match verified")

    hd_path = BASE / "human_decisions.json"
    human_decisions = {}
    if hd_path.exists():
        with open(hd_path, encoding="utf-8") as f:
            hd = json.load(f)
        human_decisions = hd.get("decisions", {})
        print(f"  {len(human_decisions)} episode overrides loaded from human_decisions.json")

    # Deduplicate won snapshots by unique_id (keep latest sorted path)
    won_by_uid: dict[str, tuple[Path, dict]] = {}
    for path, snap in won_snaps:
        uid = make_unique_id(snap, path)
        if uid not in won_by_uid or str(path) > str(won_by_uid[uid][0]):
            won_by_uid[uid] = (path, snap)

    all_entries = []
    uncertain_all = []

    # Failed episodes: adjudicate
    seen_uids: dict[str, dict] = {}
    for i, (path, snap) in enumerate(failed_snaps):
        uid = make_unique_id(snap, path)
        original_id = snap.get("id", "unknown")
        ann_ep = ann_entries[i]
        entry = make_entry(snap, path, ann_ep, human_decisions)

        # Collect uncertain items for this episode
        if ann_ep is not None:
            _, _, uncertain = adjudicate(ann_ep, original_id, human_decisions, uid)
            uncertain_all.extend(uncertain)

        if uid not in seen_uids or str(path) > str(seen_uids[uid].get("_path", "")):
            entry["_path"] = str(path)
            seen_uids[uid] = entry

    # Won episodes: include with empty failure_instances
    for path, snap in won_by_uid.values():
        uid = make_unique_id(snap, path)
        if uid not in seen_uids:
            entry = make_entry(snap, path, None, human_decisions)
            entry["_path"] = str(path)
            seen_uids[uid] = entry

    all_entries = list(seen_uids.values())
    for e in all_entries:
        e.pop("_path", None)

    # Group by game_label
    by_game: dict[str, list] = defaultdict(list)
    for e in all_entries:
        by_game[e["game"]].append(e)

    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat()

    for game_label, episodes in sorted(by_game.items()):
        game_dir = output_dir / game_label
        game_dir.mkdir(parents=True, exist_ok=True)
        n_failed = sum(1 for e in episodes if not e["won"])
        n_won = sum(1 for e in episodes if e["won"])
        out = {
            "version": "1.0",
            "game": game_label,
            "created_at": now,
            "n_episodes": len(episodes),
            "episodes": episodes,
        }
        out_path = game_dir / "snapshots.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {game_label}: {len(episodes)} episodes (failed={n_failed}, won={n_won}) → {out_path}")

    combined = {
        "version": "1.0",
        "game": "group_travel_planning",
        "created_at": now,
        "n_episodes": len(all_entries),
        "episodes": all_entries,
    }
    all_path = output_dir / "all.json"
    all_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCombined: {len(all_entries)} episodes (failed={sum(1 for e in all_entries if not e['won'])}, won={sum(1 for e in all_entries if e['won'])}) → {all_path}")

    # Uncertain cases
    uc_path = output_dir / "uncertain_cases.json"
    uc_path.write_text(json.dumps({
        "created_at": now,
        "n_items": len(uncertain_all),
        "note": "Items flagged for human review. Override via human_decisions.json keyed by unique_id.",
        "items": uncertain_all,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Uncertain cases: {len(uncertain_all)} items → {uc_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(BASE.parent.parent / "final_dataset" / "group_travel_planning"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
