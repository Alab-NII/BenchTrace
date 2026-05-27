"""
plot_cascade_crossenv.py

Cross-environment stacked bar chart comparing Qwen3-32B and GPT-4.1.
Each environment shows two bars (Qwen on top, GPT below), grouped with a
dashed separator between environments. Overall is shown first.

Usage:
    conda run -n Fraud python plot_cascade_crossenv.py \
        [--jaccard_threshold 0.5] \
        [--out output/cascade_crossenv.pdf]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ENVS = [
    ("Jer.",  "JTTL",                "reflect/output"),
    ("AW",    "AlfWorld",            "reflect/output"),
    ("BAI",   "BabyAI",              "reflect/output"),
    ("SW",    "ScienceWorld",        "reflect/output"),
    ("BWS",   "BundledWebShopping",  "reflect/output"),
    ("GTP",   "GroupTravelPlanning", "reflect/output"),
]


def _path(env_dir, out_dir, model):
    return os.path.join(BASE, env_dir, out_dir, f"{model}_results.json")


def jaccard(a, b):
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - overlap
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
    """Pass if all core failures have LLM-judge score == 2 (strict correct)."""
    items = r.get("q3", [])
    if not items:
        return True
    for item in items:
        if item.get("judge", {}).get("score") != 2:
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


def run(args):
    # Load results for both models
    all_qwen, all_gpt = [], []
    env_funnels = []  # (label, qwen_funnel, gpt_funnel)

    for label, env_dir, out_dir in ENVS:
        with open(_path(env_dir, out_dir, "qwen3-32b")) as f:
            qwen_res = json.load(f)["results"]
        with open(_path(env_dir, out_dir, "gpt-4.1")) as f:
            gpt_res = json.load(f)["results"]
        all_qwen.extend(qwen_res)
        all_gpt.extend(gpt_res)
        env_funnels.append((label,
                            compute_funnel(qwen_res, args.jaccard_threshold),
                            compute_funnel(gpt_res,  args.jaccard_threshold)))

    overall_q = compute_funnel(all_qwen, args.jaccard_threshold)
    overall_g = compute_funnel(all_gpt,  args.jaccard_threshold)

    # Groups: Overall first, then each env  (visual top→bottom)
    groups = [("Overall", overall_q, overall_g)] + env_funnels

    # barh renders bottom→top, so reverse the visual order
    groups_rev = list(reversed(groups))

    # Layout parameters
    bar_h       = 0.28
    within_gap  = 0.08   # gap between Qwen and GPT bars within a group
    between_gap = 0.42   # gap between groups (dashed line goes here)

    # Compute y positions (bottom→top)
    rows = []       # (label, y_qwen, y_gpt, midpoint, q_funnel, g_funnel)
    dashed_ys = []
    cur_y = 0.0

    for i, (label, q_f, g_f) in enumerate(groups_rev):
        y_gpt  = cur_y
        y_qwen = cur_y + bar_h + within_gap
        # barh align='center': y is the bar center, so group spans
        # [y_gpt - bar_h/2, y_qwen + bar_h/2]; midpoint = (y_gpt + y_qwen) / 2
        mid = (y_gpt + y_qwen) / 2
        rows.append((label, y_qwen, y_gpt, mid, q_f, g_f))
        cur_y = y_qwen + bar_h + between_gap
        if i < len(groups_rev) - 1:
            # dashed line halfway between top of Qwen bar and bottom of next GPT bar
            dashed_ys.append(y_qwen + bar_h / 2 + between_gap / 2)

    # Drawing
    colors     = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]
    seg_labels = ["Wrong Detection (Q1)", "Wrong Localization (Q2)", "Wrong Diagnosis (Q3)", "Correct Reflection"]
    HATCH_G    = "///"

    fig_h = cur_y * 0.58 + 1.4
    fig, ax = plt.subplots(figsize=(5.5, max(5.0, fig_h)))

    lefts_q = [0.0] * len(rows)
    lefts_g = [0.0] * len(rows)

    for seg_i, color in enumerate(colors):
        is_pass = (seg_i == 3)
        for j, (label, y_q, y_g, mid, q_f, g_f) in enumerate(rows):
            val_q = q_f[seg_i]
            val_g = g_f[seg_i]

            # Qwen bar (solid)
            if is_pass:
                ax.barh(y_q, val_q, left=lefts_q[j], height=bar_h,
                        color="white", edgecolor=color, linewidth=1.6)
            else:
                ax.barh(y_q, val_q, left=lefts_q[j], height=bar_h,
                        color=color, edgecolor="white", linewidth=0.5)
            if val_q >= 0.07:
                ax.text(lefts_q[j] + val_q / 2, y_q, f"{val_q:.0%}",
                        ha="center", va="center", fontsize=6.8, color="black",
                        fontweight="bold")
            lefts_q[j] += val_q

            # GPT-4.1 bar (hatched)
            if is_pass:
                ax.barh(y_g, val_g, left=lefts_g[j], height=bar_h,
                        color="white", edgecolor=color, linewidth=1.6,
                        hatch=HATCH_G)
            else:
                ax.barh(y_g, val_g, left=lefts_g[j], height=bar_h,
                        color=color, edgecolor="white", linewidth=0.5,
                        hatch=HATCH_G)
            if val_g >= 0.07:
                ax.text(lefts_g[j] + val_g / 2, y_g, f"{val_g:.0%}",
                        ha="center", va="center", fontsize=6.8, color="black",
                        fontweight="bold")
            lefts_g[j] += val_g

    # Dashed separators between groups
    for dy in dashed_ys:
        ax.axhline(y=dy, color="#bbbbbb", linewidth=0.8, linestyle="--")

    # Y-axis: env name at group midpoint
    ax.set_yticks([r[3] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)

    # Small "Q" / "G" labels to the right of each bar (y = bar center)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for label, y_q, y_g, mid, q_f, g_f in rows:
        ax.text(1.015, y_q, "Q", va="center", ha="left",
                fontsize=6.5, color="#333333", transform=trans)
        ax.text(1.015, y_g, "G", va="center", ha="left",
                fontsize=6.5, color="#888888", transform=trans)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of Episodes", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    # Legend: funnel segments + model indicators
    seg_handles = [mpatches.Patch(color=colors[i], label=seg_labels[i])
                   for i in range(3)]
    seg_handles += [mpatches.Patch(facecolor="white", edgecolor=colors[3],
                                   linewidth=1.6, label=seg_labels[3])]
    model_handles = [
        mpatches.Patch(facecolor="#777777", label="Qwen3-32B"),
        mpatches.Patch(facecolor="#777777", hatch=HATCH_G, edgecolor="white",
                       label="GPT-4.1"),
    ]
    fig.legend(handles=seg_handles + model_handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, fontsize=8.2, frameon=False)

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jaccard_threshold", type=float, default=0.6)
    parser.add_argument("--out", default="output/cascade_crossenv.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
