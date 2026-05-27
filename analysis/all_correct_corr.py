import math, sys
from pathlib import Path
from scipy.stats import norm
from collections import defaultdict

sys.path.insert(0, "/home/jiahao_huang/ROGUE")
from correlation_analysis import TASK_CONFIG, GPT41_SUFFIX, build_reflection_index, build_task_index, load_results_for_task
from score_evol_eval import compute_avoidance, load_dataset

Q2_THR = 0.9

def z_test(n1, p1, n2, p2):
    p_pool = (p1*n1 + p2*n2) / (n1 + n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    if se == 0: return float('nan')
    return 2 * norm.sf(abs((p1-p2)/se))

ROOT = Path("/home/jiahao_huang/ROGUE")
all_by_task = defaultdict(list)

for task_name, cfg in TASK_CONFIG.items():
    ds_path = ROOT / "final_dataset" / cfg["dataset_dir"] / "all.json"
    if not ds_path.exists(): continue
    ep_idx = load_dataset(ds_path)
    reflect = {}
    for mk, fn in [("qwen3-32b","qwen3-32b_results.json"),("gpt-4.1","gpt-4.1_results.json")]:
        rp = ROOT / cfg["reflect_dir"] / fn
        if rp.exists(): reflect[mk] = build_reflection_index(rp)
    ti = build_task_index(task_name, cfg)
    for bl, results in load_results_for_task(cfg).items():
        refl = reflect.get("gpt-4.1" if bl.endswith("_gpt41") else "qwen3-32b", {})
        for r in results:
            sid = ti.get(r.get("task_id",""))
            if not sid: continue
            sc = refl.get(sid)
            if not sc: continue
            av = compute_avoidance(r, ep_idx, cfg["use_location"], task_name)
            if av is None: continue
            if r["target_failure_instance"]["type"].split("/")[0] != "operation": continue
            all_correct = sc["q1_ok"] and (sc["q2_recall"] >= Q2_THR) and sc["q3_desc_ok"]
            all_by_task[task_name].append({"avoid": av, "all_correct": all_correct})

print(f"{'Task':<26} {'n':>5}  {'all_correct':>5}  {'FAR|correct':>12}  {'FAR|wrong':>10}  {'diff':>6}  {'p':>8}")
print("-" * 90)

all_recs = []
for task_name in TASK_CONFIG:
    recs = all_by_task.get(task_name, [])
    if not recs: continue
    all_recs.extend(recs)
    c = [r["avoid"] for r in recs if r["all_correct"]]
    w = [r["avoid"] for r in recs if not r["all_correct"]]
    if not c or not w:
        print(f"  {task_name:<24} {len(recs):>5}  {len(c):>5}  —")
        continue
    p = z_test(len(c), sum(c)/len(c), len(w), sum(w)/len(w))
    diff = sum(c)/len(c) - sum(w)/len(w)
    print(f"  {task_name:<24} {len(recs):>5}  {len(c):>5}  {sum(c)/len(c):>12.3f}  {sum(w)/len(w):>10.3f}  {diff:>+6.3f}  {p:>8.4f}")

print("-" * 90)
c = [r["avoid"] for r in all_recs if r["all_correct"]]
w = [r["avoid"] for r in all_recs if not r["all_correct"]]
p = z_test(len(c), sum(c)/len(c), len(w), sum(w)/len(w))
diff = sum(c)/len(c) - sum(w)/len(w)
print(f"  {'ALL':<24} {len(all_recs):>5}  {len(c):>5}  {sum(c)/len(c):>12.3f}  {sum(w)/len(w):>10.3f}  {diff:>+6.3f}  {p:>8.4f}")
