import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "output"

def iou(a, b):
    a0, a1 = a; b0, b1 = b
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0

games = ["detective", "library", "zork1", "zork3", "balances", "temple"]
all_ious = []

for game in games:
    path = BASE / f"{game}_ai_annotations.json"
    if not path.exists():
        continue
    data = json.load(open(path))
    for ep in data["snapshots"]:
        if not ep:
            continue
        fa_list = ep.get("claude") or []
        fb_list = ep.get("gemini") or []
        used_b = set()
        for fa in fa_list:
            wa = fa.get("where", [0, 0])
            best_iou, best_idx = 0.0, -1
            for j, fb in enumerate(fb_list):
                if j in used_b:
                    continue
                s = iou(wa, fb.get("where", [0, 0]))
                if s > best_iou:
                    best_iou, best_idx = s, j
            if best_iou >= 0.3:
                all_ious.append((game, best_iou))
                used_b.add(best_idx)

print(f"Total matched pairs: {len(all_ious)}\n")
buckets = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
print(f"{'IoU range':<15} {'count':>6} {'%':>7}")
print("-" * 32)
for lo, hi in buckets:
    cnt = sum(1 for _, v in all_ious if lo <= v < hi)
    label = f"[{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"[{lo:.1f}, 1.0]"
    print(f"{label:<15} {cnt:>6} {cnt / len(all_ious) * 100:>6.1f}%")

print(f"\nPer-game  matched  total_claude  IoU<0.5  (% of matched)")
print("-" * 55)
for game in games:
    g_ious = [v for g, v in all_ious if g == game]
    if not g_ious:
        continue
    path = BASE / f"{game}_ai_annotations.json"
    data = json.load(open(path))
    total_c = sum(len(ep.get("claude") or []) for ep in data["snapshots"] if ep)
    low = sum(1 for v in g_ious if v < 0.5)
    print(f"{game:<12} {len(g_ious):>7}  {total_c:>12}  {low:>7}  ({low / len(g_ious) * 100:.0f}%)")
