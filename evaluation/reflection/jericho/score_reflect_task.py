"""
score_reflect_task.py

Computes Reflection Task metrics from the output of run_reflect_task.py.

Metrics:
  Q1 (Detection):
    - accuracy: fraction of episodes where model answered "yes"

  Q2 (Localization):
    - similarity: mean over model ranges of max Jaccard to any GT core failure range
    - recall:     mean over GT core failure ranges of max Jaccard to any model range

  Q3 (Diagnosis):
    - type_accuracy: exact match of failure_type vs GT
    - description_f1: token-level F1 between description and GT diagnosis
    - llm_judge: fraction judged as describing the same failure by LLM

Usage:
    conda run -n Fraud python score_reflect_task.py \\
        --results output/claude-sonnet-4-6_results.json \\
        [--judge_model claude-sonnet-4-6] \\
        [--skip_llm_judge]
"""

import argparse
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ── IoU / Jaccard ─────────────────────────────────────────────────────────────

def jaccard(a: list[int], b: list[int]) -> float:
    a0, a1 = a[0], a[1]
    b0, b1 = b[0], b[1]
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


# ── Token F1 ──────────────────────────────────────────────────────────────────

def token_f1(pred: str, gold: str) -> float:
    pred_tokens = re.findall(r"\w+", pred.lower())
    gold_tokens = re.findall(r"\w+", gold.lower())
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = defaultdict(int)
    for t in pred_tokens:
        pred_set[t] += 1
    gold_set = defaultdict(int)
    for t in gold_tokens:
        gold_set[t] += 1
    common = sum(min(pred_set[t], gold_set[t]) for t in pred_set if t in gold_set)
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── LLM Judge ─────────────────────────────────────────────────────────────────

LLM_JUDGE_SYSTEM = """You are a strict evaluator. You will be shown two one-sentence descriptions of an agent's failure in a text adventure game. Determine whether they describe the same underlying failure. Minor differences in wording are acceptable; what matters is whether the core failure being described is the same.

Respond with JSON only: {"same": true} or {"same": false}"""


def llm_judge(pred_desc: str, gt_desc: str, judge_model: str) -> bool | None:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user = f"Description A: {pred_desc}\n\nDescription B: {gt_desc}"
    try:
        msg = client.messages.create(
            model=judge_model,
            max_tokens=64,
            system=LLM_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = msg.content[0].text
        match = re.search(r'"same"\s*:\s*(true|false)', text)
        if match:
            return match.group(1) == "true"
    except Exception as e:
        print(f"  LLM judge error: {e}")
    return None


# ── Per-question scoring ──────────────────────────────────────────────────────

def score_q1(results: list[dict]) -> dict:
    total = len(results)
    yes_count = sum(1 for r in results if r["q1"].get("parsed") == "yes")
    parse_fail = sum(1 for r in results if r["q1"].get("parsed") is None)
    return {
        "n": total,
        "accuracy": yes_count / total if total else 0.0,
        "parse_failures": parse_fail,
    }


def score_q2(results: list[dict]) -> dict:
    sim_scores = []
    rec_scores = []
    skipped = 0

    for r in results:
        from_snapshot = r.get("_gt_core_ranges")  # injected below
        gt_ranges = from_snapshot
        pred_ranges = r["q2"].get("parsed")

        if not gt_ranges:
            skipped += 1
            continue

        if not pred_ranges:
            # model failed to produce ranges → 0 for both
            sim_scores.append(0.0)
            rec_scores.append(0.0)
            continue

        # Similarity (precision direction): for each pred range, best match in GT
        sim = sum(
            max(jaccard(pr, gr) for gr in gt_ranges)
            for pr in pred_ranges
        ) / len(pred_ranges)
        sim_scores.append(sim)

        # Recall: for each GT range, best match in pred
        rec = sum(
            max(jaccard(gr, pr) for pr in pred_ranges)
            for gr in gt_ranges
        ) / len(gt_ranges)
        rec_scores.append(rec)

    n = len(sim_scores)
    return {
        "n": n,
        "skipped_no_gt": skipped,
        "similarity": sum(sim_scores) / n if n else 0.0,
        "recall": sum(rec_scores) / n if n else 0.0,
    }


def score_q3(results: list[dict], judge_model: str | None, workers: int) -> dict:
    # Flatten all Q3 items
    items = []
    for r in results:
        for q3 in r.get("q3", []):
            items.append(q3)

    if not items:
        return {"n": 0}

    type_correct = []
    f1_scores = []
    judge_scores = []
    parse_fails = 0

    judge_items = []  # (idx, pred_desc, gt_desc) for LLM judge

    for i, item in enumerate(items):
        parsed = item.get("parsed")
        if not parsed:
            parse_fails += 1
            type_correct.append(0)
            f1_scores.append(0.0)
            continue

        # Type accuracy
        type_correct.append(int(parsed["failure_type"] == item["gt_type"]))

        # Description F1
        f1_scores.append(token_f1(parsed["description"], item["gt_diagnosis"]))

        if judge_model:
            judge_items.append((i, parsed["description"], item["gt_diagnosis"]))

    # LLM judge (parallel)
    if judge_model and judge_items:
        judge_results = [None] * len(items)
        lock = threading.Lock()
        completed = 0

        def run_judge(idx, pred, gt):
            nonlocal completed
            result = llm_judge(pred, gt, judge_model)
            with lock:
                judge_results[idx] = result
                completed += 1
                if completed % 10 == 0:
                    print(f"  LLM judge: {completed}/{len(judge_items)}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_judge, idx, pred, gt)
                       for idx, pred, gt in judge_items]
            for f in futures:
                f.result()

        judge_scores = [r for r in judge_results if r is not None]
    else:
        judge_scores = []

    n = len(items)
    out = {
        "n": n,
        "parse_failures": parse_fails,
        "type_accuracy": sum(type_correct) / n if n else 0.0,
        "description_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }
    if judge_scores:
        out["llm_judge_accuracy"] = sum(judge_scores) / len(judge_scores)
        out["llm_judge_n"] = len(judge_scores)
    return out


# ── Per-game breakdown ────────────────────────────────────────────────────────

def breakdown_by_game(results: list[dict], score_fn, label: str):
    by_game = defaultdict(list)
    for r in results:
        by_game[r["game"]].append(r)
    breakdown = {}
    for game, game_results in sorted(by_game.items()):
        breakdown[game] = score_fn(game_results)
    return breakdown


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} episode results | model: {data['model']}")

    # Inject GT core ranges into results for Q2 scoring
    for r in results:
        # GT core ranges are embedded in q3 items
        r["_gt_core_ranges"] = [item["where"] for item in r.get("q3", [])]

    judge_model = None if args.skip_llm_judge else args.judge_model

    print("\n── Q1 (Detection) ──")
    q1 = score_q1(results)
    print(f"  accuracy: {q1['accuracy']:.3f}  (n={q1['n']}, parse_fail={q1['parse_failures']})")

    print("\n── Q2 (Localization) ──")
    q2 = score_q2(results)
    print(f"  similarity: {q2['similarity']:.3f}  recall: {q2['recall']:.3f}  "
          f"(n={q2['n']}, skipped_no_gt={q2['skipped_no_gt']})")

    print("\n── Q3 (Diagnosis) ──")
    q3 = score_q3(results, judge_model, args.workers)
    print(f"  type_accuracy: {q3['type_accuracy']:.3f}  "
          f"description_f1: {q3['description_f1']:.3f}  "
          f"(n={q3['n']}, parse_fail={q3.get('parse_failures', 0)})")
    if "llm_judge_accuracy" in q3:
        print(f"  llm_judge_accuracy: {q3['llm_judge_accuracy']:.3f}  "
              f"(n={q3['llm_judge_n']})")

    # Per-game breakdown
    print("\n── Per-game Q1 ──")
    for game, s in breakdown_by_game(results, score_q1, "q1").items():
        print(f"  {game}: accuracy={s['accuracy']:.3f}")

    print("\n── Per-game Q2 ──")
    for game, rs in defaultdict(list, {r["game"]: [] for r in results}).items():
        pass
    by_game = defaultdict(list)
    for r in results:
        by_game[r["game"]].append(r)
    for game in sorted(by_game):
        s = score_q2(by_game[game])
        print(f"  {game}: sim={s['similarity']:.3f}  rec={s['recall']:.3f}")

    # Save scores
    score_path = Path(args.results).with_suffix(".scores.json")
    summary = {
        "model": data["model"],
        "api": data["api"],
        "games": data["games"],
        "q1": q1,
        "q2": q2,
        "q3": q3,
    }
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nScores saved to: {score_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results JSON from run_reflect_task.py")
    parser.add_argument("--judge_model", default="claude-sonnet-4-6",
                        help="Model to use as LLM judge for Q3")
    parser.add_argument("--skip_llm_judge", action="store_true",
                        help="Skip LLM-as-judge scoring for Q3")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent LLM judge calls (default: 8)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
