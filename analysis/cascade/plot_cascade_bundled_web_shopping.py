"""
plot_cascade.py

Stacked bar chart of Q1/Q2/Q3 funnel analysis for Qwen3-32B and GPT-4.1.

Usage:
    conda run -n Fraud python plot_cascade.py \
        --qwen output/qwen3-32b_results.json \
        --gpt output/gpt-4.1_results.json \
        [--jaccard_threshold 0.5] \
        [--out output/cascade.pdf]
"""

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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
        return True
    pred_ranges = r["q2"].get("parsed") or []
    if not pred_ranges:
        return False
    return any(jaccard(pr, gr) >= threshold for pr in pred_ranges for gr in gt_ranges)


def q3_pass(r):
    items = r.get("q3", [])
    if not items:
        return True
    for item in items:
        score = item.get("judge", {}).get("score")
        if score is None or score < 2:
            return False
    return True


def compute_funnel(results, threshold):
    total = len(results)
    fail_q1 = fail_q2 = fail_q3 = pass_all = 0
    for r in results:
        if not q1_pass(r):
            fail_q1 += 1
        elif not q2_pass(r, threshold):
            fail_q2 += 1
        elif not q3_pass(r):
            fail_q3 += 1
        else:
            pass_all += 1
    return [fail_q1 / total, fail_q2 / total, fail_q3 / total, pass_all / total]


GAMES = ["balances", "detective", "library", "temple", "zork1", "zork3"]
GAME_LABELS = ["Balances", "Detective", "Library", "Temple", "Zork1", "Zork3"]


def compute_funnel_by_game(results, threshold):
    by_game = defaultdict(list)
    for r in results:
        by_game[r["game"]].append(r)
    overall = compute_funnel(results, threshold)
    per_game = {g: compute_funnel(by_game[g], threshold) for g in GAMES}
    return overall, per_game


def draw_panel(ax, overall, per_game, title, colors, seg_labels, show_ylabel=True):
    # Reverse game order so alphabetical top-to-bottom (barh renders bottom-up)
    games_reversed = list(reversed(GAMES))
    labels_reversed = list(reversed(GAME_LABELS))

    # Custom y positions: per-game rows at 0..5, small gap, Overall at top
    game_ys = np.arange(len(GAMES), dtype=float)   # 0,1,2,3,4,5
    overall_y = len(GAMES) + 0.7                    # 6.7 — small gap above games

    row_ys = list(game_ys) + [overall_y]
    row_labels = labels_reversed + ["Overall"]
    row_data = [per_game[g] for g in games_reversed] + [overall]
    bar_heights = [0.35] * len(GAMES) + [0.7]

    lefts = np.zeros(len(row_ys))

    for i, color in enumerate(colors):
        vals = np.array([d[i] for d in row_data])
        for j, (y_pos, val, left, bh) in enumerate(zip(row_ys, vals, lefts, bar_heights)):
            ax.barh(y_pos, val, left=left, height=bh,
                    color=color, edgecolor="white", linewidth=0.7)
            if val > 0.07:
                fontsize = 9 if j == len(GAMES) else 7.5
                ax.text(left + val / 2, y_pos, f"{val:.0%}",
                        ha="center", va="center", fontsize=fontsize,
                        color="white", fontweight="bold")
        lefts = lefts + vals

    # Separator line between Overall and per-game rows
    ax.axhline(y=overall_y - 0.7, color="#aaaaaa", linewidth=0.8, linestyle="--")

    ax.set_yticks(row_ys)
    if show_ylabel:
        ax.set_yticklabels(row_labels, fontsize=9)
    else:
        ax.set_yticklabels([""] * len(row_ys))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of Episodes", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def run(args):
    with open(args.qwen) as f:
        qwen_data = json.load(f)
    with open(args.gpt) as f:
        gpt_data = json.load(f)

    qwen_overall, qwen_per_game = compute_funnel_by_game(qwen_data["results"], args.jaccard_threshold)
    gpt_overall, gpt_per_game = compute_funnel_by_game(gpt_data["results"], args.jaccard_threshold)

    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
    seg_labels = ["Fail Q1 (Detection)", "Fail Q2 (Localization)", "Fail Q3 (Diagnosis)", "Pass All"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    draw_panel(ax1, qwen_overall, qwen_per_game, "Qwen3-32B", colors, seg_labels, show_ylabel=True)
    draw_panel(ax2, gpt_overall, gpt_per_game, "GPT-4.1", colors, seg_labels, show_ylabel=False)

    # Shared legend at bottom
    handles = [mpatches.Patch(color=colors[i], label=seg_labels[i]) for i in range(4)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=4, fontsize=8.5, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", default="output/qwen3-32b_results.json")
    parser.add_argument("--gpt", default="output/gpt-4.1_results.json")
    parser.add_argument("--jaccard_threshold", type=float, default=0.5)
    parser.add_argument("--out", default="output/cascade.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
