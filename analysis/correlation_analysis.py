"""
correlation_analysis.py

Correlation between reflection evaluation quality (on signal snapshots) and
evolution evaluation avoidance rate (on test snapshots) — all tasks.

Signal snapshot = evolution_snapshots[0] (the rest are noise snapshots).

Model matching:
  - Baselines without _gpt41  →  qwen3-32b reflection scores
  - Baselines with    _gpt41  →  gpt-4.1   reflection scores

Usage:
    conda run -n Fraud python correlation_analysis.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from score_evol_eval import (  # noqa: E402
    compute_avoidance,
    load_dataset,
    DEFAULT_SUBTASKS,
)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "jericho": {
        "result_dir":  "Jericho",
        "dataset_dir": "jericho",
        "reflect_dir": "JTTL/reflect/output",
        "subtasks":    DEFAULT_SUBTASKS["jericho"],
        "use_location": True,
    },
    "alfworld": {
        "result_dir":  "AlfWorld",
        "dataset_dir": "alfworld",
        "reflect_dir": "AlfWorld/reflect/output",
        "subtasks":    DEFAULT_SUBTASKS["alfworld"],
        "use_location": True,
    },
    "babyai": {
        "result_dir":  "BabyAI",
        "dataset_dir": "babyai",
        "reflect_dir": "BabyAI/reflect/output",
        "subtasks":    DEFAULT_SUBTASKS["babyai"],
        "use_location": True,
    },
    "scienceworld": {
        "result_dir":  "ScienceWorld",
        "dataset_dir": "scienceworld",
        "reflect_dir": "ScienceWorld/reflect/output",
        "subtasks":    DEFAULT_SUBTASKS["scienceworld"],
        "use_location": True,
    },
    "bundled_web_shopping": {
        "result_dir":  "BundledWebShopping",
        "dataset_dir": "bundled_web_shopping",
        "reflect_dir": "BundledWebShopping/reflect/output",
        "subtasks":    DEFAULT_SUBTASKS["bundled_web_shopping"],
        "use_location": False,
    },
    "group_travel_planning": {
        "result_dir":  "GroupTravelPlanning",
        "dataset_dir": "group_travel_planning",
        "reflect_dir": "GroupTravelPlanning/reflect/output",
        "subtasks":    DEFAULT_SUBTASKS["group_travel_planning"],
        "use_location": False,
    },
}

GPT41_SUFFIX        = "_gpt41"
Q2_RECALL_THRESHOLD = 0.5
Q3_DESC_THRESHOLD   = 2

# ---------------------------------------------------------------------------
# Reflection scoring
# ---------------------------------------------------------------------------

def _jaccard(a: list, b: list) -> float:
    sa = set(range(a[0], a[1] + 1))
    sb = set(range(b[0], b[1] + 1))
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _q2_recall(pred_ranges: list, gt_ranges: list) -> float:
    if not pred_ranges or not gt_ranges:
        return 0.0
    return sum(
        max(_jaccard(gr, pr) for pr in pred_ranges)
        for gr in gt_ranges
    ) / len(gt_ranges)


def build_reflection_index(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    index: dict[str, dict] = {}
    for r in data["results"]:
        sid       = r["id"]
        q1_ok     = r["q1"].get("parsed") == "yes"
        gt_ranges = [item["where"] for item in r.get("q3", [])]
        pred_r    = r["q2"].get("parsed") or []
        recall    = _q2_recall(pred_r, gt_ranges)
        q2_ok     = recall >= Q2_RECALL_THRESHOLD
        q3_items  = r.get("q3", [])
        q3_type_ok = any(
            (item.get("parsed") or {}).get("failure_type") == item.get("gt_type")
            for item in q3_items
        )
        def _desc_score(item):
            s = (item.get("judge") or {}).get("score")
            return s if s is not None else 0
        q3_desc_ok = any(_desc_score(item) >= Q3_DESC_THRESHOLD for item in q3_items)
        index[sid] = {
            "q1_ok":      q1_ok,
            "q2_recall":  recall,
            "q2_ok":      q2_ok,
            "q3_type_ok": q3_type_ok,
            "q3_desc_ok": q3_desc_ok,
        }
    return index


# ---------------------------------------------------------------------------
# Task index: task_id → signal_snapshot_id
# ---------------------------------------------------------------------------

def build_task_index(task_name: str, cfg: dict) -> dict[str, str]:
    dataset_dir = ROOT / "final_dataset" / cfg["dataset_dir"]
    index: dict[str, str] = {}
    for subtask in cfg["subtasks"]:
        ee_path = dataset_dir / subtask / "evolution_evaluation.json"
        if not ee_path.exists():
            continue
        with open(ee_path) as f:
            data = json.load(f)
        for task in data["tasks"]:
            index[task["id"]] = task["evolution_snapshots"][0]
    return index


# ---------------------------------------------------------------------------
# Load evolution results
# ---------------------------------------------------------------------------

def load_results_for_task(cfg: dict) -> dict[str, list[dict]]:
    results_dir = ROOT / "main_result" / cfg["result_dir"]
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    for subtask in cfg["subtasks"]:
        subtask_dir = results_dir / subtask
        if not subtask_dir.exists():
            continue
        for bl_dir in sorted(subtask_dir.iterdir()):
            if not bl_dir.is_dir():
                continue
            rpath = bl_dir / "results.json"
            if not rpath.exists():
                continue
            with open(rpath) as f:
                data = json.load(f)
            by_baseline[bl_dir.name].extend(data.get("results", []))
    return dict(by_baseline)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

FUNNEL_STEPS  = ["q1_ok", "q2_ok", "q3_type_ok", "q3_desc_ok"]
FUNNEL_LABELS = ["Q1 whether", "Q2 where   ", "Q3 type    ", "Q3 desc    "]


def conditional_table(records: list[dict]) -> list[dict]:
    rows = []
    for step, label in zip(FUNNEL_STEPS, FUNNEL_LABELS):
        correct = [r["avoid"] for r in records if r[step] is True]
        wrong   = [r["avoid"] for r in records if r[step] is False]
        avg_c = sum(correct) / len(correct) if correct else None
        avg_w = sum(wrong)   / len(wrong)   if wrong   else None
        diff  = (avg_c - avg_w) if (avg_c is not None and avg_w is not None) else None
        rows.append({"step": label, "n_correct": len(correct), "avg_correct": avg_c,
                     "n_wrong": len(wrong), "avg_wrong": avg_w, "diff": diff})
    return rows


def print_conditional_table(records: list[dict]):
    rows = conditional_table(records)
    print(f"  {'Step':<14} {'avoid|correct':>14} {'avoid|wrong':>13} {'diff':>7}")
    print("  " + "-" * 52)
    for row in rows:
        ac = f"{row['avg_correct']:.3f} (n={row['n_correct']:4d})" if row['avg_correct'] is not None else f"  —    (n={row['n_correct']:4d})"
        aw = f"{row['avg_wrong']:.3f} (n={row['n_wrong']:4d})"     if row['avg_wrong']   is not None else f"  —    (n={row['n_wrong']:4d})"
        df = f"{row['diff']:+.3f}" if row['diff'] is not None else "   —  "
        print(f"  {row['step']:<14}  {ac}  {aw}  {df}")


def funnel_breakdown(records: list[dict]) -> list[dict]:
    rows = []
    cohort = records
    for step, label in zip(FUNNEL_STEPS, FUNNEL_LABELS):
        n_total   = len(cohort)
        n_correct = sum(1 for r in cohort if r[step] is True)
        n_wrong   = n_total - n_correct
        rows.append({"step": label, "n_total": n_total,
                     "n_correct": n_correct, "n_wrong": n_wrong})
        cohort = [r for r in cohort if r[step] is True]
    rows.append({"step": "All correct", "n_total": len(records),
                 "n_correct": len(cohort), "n_wrong": 0})
    return rows


def print_funnel(records: list[dict], title: str):
    n = len(records)
    if n == 0:
        print(f"  {title}: (no records)")
        return
    print(f"  {title} (n={n})")
    for row in funnel_breakdown(records):
        pct = 100 * row["n_correct"] / row["n_total"] if row["n_total"] else 0
        broken_pct = 100 * row["n_wrong"] / row["n_total"] if row["n_total"] else 0
        if row["step"] == "All correct":
            print(f"    → All 4 correct: {row['n_correct']}/{n} ({pct:.1f}%)")
        else:
            print(f"    {row['step']}: pass={row['n_correct']}/{row['n_total']} ({pct:.1f}%)  "
                  f"break_here={row['n_wrong']} ({broken_pct:.1f}%)")


def print_section(records: list[dict], title: str):
    n = len(records)
    if n == 0:
        return
    avoid_rate = sum(r["avoid"] for r in records) / n
    print(f"\n{'─'*60}")
    print(f"  {title}  (n={n}, avoid={avoid_rate:.3f})")
    print(f"{'─'*60}")
    print_conditional_table(records)
    avoid0 = [r for r in records if r["avoid"] == 0.0]
    avoid1 = [r for r in records if r["avoid"] == 1.0]
    print()
    print_funnel(avoid0, "Funnel  avoid=0")
    print_funnel(avoid1, "Funnel  avoid=1 (reference)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_records: list[dict] = []

    for task_name, cfg in TASK_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"  Task: {task_name}")
        print(f"{'='*60}")

        # Episode index
        ds_path = ROOT / "final_dataset" / cfg["dataset_dir"] / "all.json"
        if not ds_path.exists():
            print(f"  [skip] dataset not found: {ds_path}")
            continue
        episode_index = load_dataset(ds_path)
        print(f"  episodes: {len(episode_index)}")

        # Reflection index
        reflect: dict[str, dict] = {}
        for model_key, fname in [("qwen3-32b", "qwen3-32b_results.json"),
                                  ("gpt-4.1",   "gpt-4.1_results.json")]:
            rpath = ROOT / cfg["reflect_dir"] / fname
            if rpath.exists():
                reflect[model_key] = build_reflection_index(rpath)
        print(f"  reflect models: {list(reflect)}")

        # Task index
        task_index = build_task_index(task_name, cfg)
        print(f"  eval tasks: {len(task_index)}")

        # Evolution results
        by_baseline = load_results_for_task(cfg)
        total = sum(len(v) for v in by_baseline.values())
        print(f"  baselines: {sorted(by_baseline)}  ({total} results)")

        # Build records
        task_records: list[dict] = []
        skipped = defaultdict(int)

        for baseline, results in by_baseline.items():
            model = "gpt-4.1" if baseline.endswith(GPT41_SUFFIX) else "qwen3-32b"
            refl  = reflect.get(model, {})

            for r in results:
                tid       = r.get("task_id", "")
                signal_id = task_index.get(tid)
                if signal_id is None:
                    skipped["no_task"] += 1
                    continue
                scores = refl.get(signal_id)
                if scores is None:
                    skipped["no_refl"] += 1
                    continue
                avoid = compute_avoidance(r, episode_index,
                                          cfg["use_location"], task_name)
                if avoid is None:
                    skipped["no_avoid"] += 1
                    continue

                cat = r["target_failure_instance"]["type"].split("/")[0]
                task_records.append({
                    "task":        task_name,
                    "baseline":    baseline,
                    "task_id":     tid,
                    "signal_id":   signal_id,
                    "avoid":       avoid,
                    "failure_cat": cat,
                    **{k: scores[k] for k in
                       ["q1_ok", "q2_ok", "q3_type_ok", "q3_desc_ok"]},
                })

        print(f"  records: {len(task_records)}  skipped: {dict(skipped)}")

        # Per-task per-category analysis
        for cat in ["ALL", "operation", "strategy"]:
            subset = task_records if cat == "ALL" else \
                     [r for r in task_records if r["failure_cat"] == cat]
            print_section(subset, f"{task_name} | {cat}")

        all_records.extend(task_records)

    # ── Aggregate across all tasks ──────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print(f"  AGGREGATE — ALL TASKS  (n={len(all_records)})")
    print(f"{'#'*60}")

    for cat in ["ALL", "operation", "strategy"]:
        subset = all_records if cat == "ALL" else \
                 [r for r in all_records if r["failure_cat"] == cat]
        print_section(subset, f"ALL TASKS | {cat}")

    # ── Per-task summary table ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Per-task summary (operation failures only)")
    print(f"{'='*60}")
    header = f"  {'Task':<24} {'n':>5}  {'avoid':>6}  {'Q1ok':>6}  {'Q2ok':>6}  {'Q3t_ok':>7}  {'Q3d_ok':>7}  {'Δ(Q3t)':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for task_name in TASK_CONFIG:
        sub = [r for r in all_records
               if r["task"] == task_name and r["failure_cat"] == "operation"]
        if not sub:
            continue
        n = len(sub)
        def rate(k):
            vals = [r[k] for r in sub if r[k] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        # delta for q3_type
        ct = [r["avoid"] for r in sub if r["q3_type_ok"] is True]
        cw = [r["avoid"] for r in sub if r["q3_type_ok"] is False]
        delta = (sum(ct)/len(ct) - sum(cw)/len(cw)) if ct and cw else float("nan")
        print(
            f"  {task_name:<24} {n:>5}  {rate('avoid'):>6.3f}  "
            f"{rate('q1_ok'):>6.3f}  {rate('q2_ok'):>6.3f}  "
            f"{rate('q3_type_ok'):>7.3f}  {rate('q3_desc_ok'):>7.3f}  "
            f"{delta:>+7.3f}"
        )


if __name__ == "__main__":
    main()
