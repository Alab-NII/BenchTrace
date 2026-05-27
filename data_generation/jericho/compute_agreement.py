"""
compute_agreement.py

Computes inter-annotator agreement between Claude and Gemini annotations.

Stage 1: where overlap — for each episode, compute IoU between step ranges
         to determine whether the two models identified the same failures.
Stage 2: classification agreement — for matched failure pairs, compute
         Cohen's Kappa separately for 'type' and 'tier'.

Usage:
    conda run -n Fraud python compute_agreement.py \
        --annotation_file output/zork1/zork1_ai_annotations.json \
        [--iou_threshold 0.3]
"""

import argparse
import json
from collections import Counter
from pathlib import Path


# ── IoU matching ─────────────────────────────────────────────────────────────

def iou(range_a: list[int], range_b: list[int]) -> float:
    """Compute IoU between two step ranges [first, last] (inclusive)."""
    a0, a1 = range_a
    b0, b1 = range_b
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


def match_failures(
    failures_a: list[dict],
    failures_b: list[dict],
    threshold: float,
) -> tuple[list[tuple], list[dict], list[dict]]:
    """
    Greedily match failures from model A to model B by IoU on 'where'.
    Returns:
        matched: list of (failure_a, failure_b) pairs
        unmatched_a: failures in A with no match in B
        unmatched_b: failures in B with no match in A
    """
    used_b = set()
    matched = []

    for fa in failures_a:
        where_a = fa.get("where", [0, 0])
        best_iou, best_idx = 0.0, -1
        for j, fb in enumerate(failures_b):
            if j in used_b:
                continue
            score = iou(where_a, fb.get("where", [0, 0]))
            if score > best_iou:
                best_iou, best_idx = score, j
        if best_iou >= threshold:
            matched.append((fa, failures_b[best_idx]))
            used_b.add(best_idx)

    unmatched_a = [fa for i, fa in enumerate(failures_a)
                   if not any(fa is m[0] for m in matched)]
    unmatched_b = [fb for j, fb in enumerate(failures_b) if j not in used_b]
    return matched, unmatched_a, unmatched_b


# ── Cohen's Kappa ─────────────────────────────────────────────────────────────

def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's Kappa for two lists of categorical labels."""
    assert len(labels_a) == len(labels_b), "Label lists must have equal length"
    n = len(labels_a)
    if n == 0:
        return float("nan")

    categories = sorted(set(labels_a) | set(labels_b))
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)

    # Observed agreement
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement
    p_e = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ── Main ──────────────────────────────────────────────────────────────────────

def jaccard(labels_a: list, labels_b: list, positive: str = "core") -> float:
    """
    Jaccard similarity for a binary label among matched pairs.
    both / (both + one_a + one_b), where 'both' = both equal positive,
    one_a/one_b = one is positive and the other is not.
    """
    both = sum(1 for a, b in zip(labels_a, labels_b) if a == positive and b == positive)
    one  = sum(1 for a, b in zip(labels_a, labels_b) if (a == positive) != (b == positive))
    return both / (both + one) if (both + one) > 0 else float("nan")


def _f1(matched: int, total_a: int, total_b: int) -> float:
    precision = matched / total_b if total_b else 0.0
    recall    = matched / total_a if total_a else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


def compute(annotation_file: Path, iou_threshold: float):
    with open(annotation_file, encoding="utf-8") as f:
        data = json.load(f)

    episodes = data["snapshots"]

    # Overall accumulators
    ov_total_a, ov_total_b, ov_matched = 0, 0, 0
    ov_type_a, ov_type_b = [], []
    ov_tier_a, ov_tier_b = [], []

    # Core-only accumulators
    co_total_a, co_total_b, co_matched = 0, 0, 0
    co_type_a, co_type_b = [], []
    co_tier_a, co_tier_b = [], []

    per_episode = []

    for ep in episodes:
        fa_list = ep.get("claude") or []
        fb_list = ep.get("gemini") or []

        if fa_list is None or fb_list is None:
            continue

        # ── Overall matching ──────────────────────────────────────────────
        matched, unmatched_a, unmatched_b = match_failures(fa_list, fb_list, iou_threshold)

        ov_total_a += len(fa_list)
        ov_total_b += len(fb_list)
        ov_matched += len(matched)

        for fa, fb in matched:
            ov_type_a.append(fa.get("type", "unknown"))
            ov_type_b.append(fb.get("type", "unknown"))
            ov_tier_a.append(fa.get("tier", "unknown"))
            ov_tier_b.append(fb.get("tier", "unknown"))

        # ── Core-only matching ────────────────────────────────────────────
        fa_core = [f for f in fa_list if f.get("tier") == "core"]
        fb_core = [f for f in fb_list if f.get("tier") == "core"]
        matched_core, _, _ = match_failures(fa_core, fb_core, iou_threshold)

        co_total_a += len(fa_core)
        co_total_b += len(fb_core)
        co_matched += len(matched_core)

        for fa, fb in matched_core:
            co_type_a.append(fa.get("type", "unknown"))
            co_type_b.append(fb.get("type", "unknown"))
            co_tier_a.append(fa.get("tier", "unknown"))
            co_tier_b.append(fb.get("tier", "unknown"))

        per_episode.append({
            "id": ep["id"],
            "claude_failures": len(fa_list),
            "gemini_failures": len(fb_list),
            "matched": len(matched),
            "unmatched_claude": len(unmatched_a),
            "unmatched_gemini": len(unmatched_b),
            "claude_core": len(fa_core),
            "gemini_core": len(fb_core),
            "matched_core": len(matched_core),
        })

    def _round(v):
        return round(v, 4) if v == v else None  # NaN guard

    report = {
        "game": data.get("game"),
        "models": data.get("models"),
        "iou_threshold": iou_threshold,
        "n_episodes": len(per_episode),
        "core": {
            "where_f1":     _round(_f1(co_matched, co_total_a, co_total_b)),
            "type_kappa":   _round(cohens_kappa(co_type_a, co_type_b)),
            "tier_jaccard": _round(jaccard(ov_tier_a, ov_tier_b)),
        },
        "overall": {
            "where_f1":   _round(_f1(ov_matched, ov_total_a, ov_total_b)),
            "type_kappa": _round(cohens_kappa(ov_type_a, ov_type_b)),
            "tier_kappa": _round(cohens_kappa(ov_tier_a, ov_tier_b)),
        },
        "per_episode": per_episode,
    }

    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", required=True,
                        help="Path to AI annotation JSON from ai_annotate.py")
    parser.add_argument("--iou_threshold", type=float, default=0.3,
                        help="Minimum IoU to consider two failures matched (default: 0.3)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save the report JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = compute(Path(args.annotation_file), args.iou_threshold)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")
