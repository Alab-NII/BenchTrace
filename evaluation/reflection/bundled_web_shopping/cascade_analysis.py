"""
cascade_analysis.py

Funnel analysis: how many episodes fail at Q1, Q2, Q3 in sequence.

Q1 pass: model answered "yes"
Q2 pass: at least one predicted range has Jaccard >= threshold with any GT core range
Q3 pass: LLM-judge score >= 1 (partial or correct) for all core failures

Usage:
    conda run -n Fraud python cascade_analysis.py \
        --results output/qwen3-32b_results.json \
        [--jaccard_threshold 0.5]
"""

import argparse
import json
from collections import defaultdict


def jaccard(a, b):
    a0, a1 = a[0], a[1]
    b0, b1 = b[0], b[1]
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


def q1_pass(r):
    return r["q1"].get("parsed") == "yes"


def q2_pass(r, threshold):
    gt_ranges = [item["where"] for item in r.get("q3", [])]
    if not gt_ranges:
        return True  # no GT to check against, skip
    pred_ranges = r["q2"].get("parsed") or []
    if not pred_ranges:
        return False
    # pass if any predicted range matches any GT range above threshold
    return any(
        jaccard(pr, gr) >= threshold
        for pr in pred_ranges
        for gr in gt_ranges
    )


def q3_pass(r):
    items = r.get("q3", [])
    if not items:
        return True
    # pass if all core failures have judge score >= 1
    for item in items:
        judge = item.get("judge", {})
        score = judge.get("score")
        if score is None or score < 2:
            return False
    return True


def run(args):
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    model = data["model"]
    print(f"Model: {model}  |  Episodes: {len(results)}  |  Jaccard threshold: {args.jaccard_threshold}\n")

    total = len(results)
    fail_q1 = 0
    fail_q2 = 0
    fail_q3 = 0
    pass_all = 0

    by_game = defaultdict(lambda: {"total": 0, "fail_q1": 0, "fail_q2": 0, "fail_q3": 0, "pass_all": 0})

    for r in results:
        game = r["game"]
        by_game[game]["total"] += 1

        if not q1_pass(r):
            fail_q1 += 1
            by_game[game]["fail_q1"] += 1
            continue
        if not q2_pass(r, args.jaccard_threshold):
            fail_q2 += 1
            by_game[game]["fail_q2"] += 1
            continue
        if not q3_pass(r):
            fail_q3 += 1
            by_game[game]["fail_q3"] += 1
            continue
        pass_all += 1
        by_game[game]["pass_all"] += 1

    print("── Overall Funnel ──")
    print(f"  Total episodes:      {total}")
    print(f"  Fail at Q1:          {fail_q1:3d}  ({fail_q1/total:.1%})")
    print(f"  Fail at Q2:          {fail_q2:3d}  ({fail_q2/total:.1%})  [passed Q1]")
    print(f"  Fail at Q3:          {fail_q3:3d}  ({fail_q3/total:.1%})  [passed Q1+Q2]")
    print(f"  Pass all three:      {pass_all:3d}  ({pass_all/total:.1%})")

    print("\n── Per-game Funnel ──")
    print(f"  {'Game':<12} {'Total':>6} {'Fail Q1':>8} {'Fail Q2':>8} {'Fail Q3':>8} {'Pass all':>9}")
    for game in sorted(by_game):
        g = by_game[game]
        n = g["total"]
        print(f"  {game:<12} {n:>6} "
              f"{g['fail_q1']:>5} ({g['fail_q1']/n:.0%}) "
              f"{g['fail_q2']:>5} ({g['fail_q2']/n:.0%}) "
              f"{g['fail_q3']:>5} ({g['fail_q3']/n:.0%}) "
              f"{g['pass_all']:>5} ({g['pass_all']/n:.0%})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--jaccard_threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
