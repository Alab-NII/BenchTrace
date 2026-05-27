import math, sys
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, "/home/jiahao_huang/ROGUE")
from correlation_analysis import TASK_CONFIG, GPT41_SUFFIX, build_reflection_index, build_task_index, load_results_for_task
from score_evol_eval import compute_avoidance, load_dataset

def z_test(n1, p1, n2, p2):
    pp = (p1*n1+p2*n2)/(n1+n2)
    se = math.sqrt(pp*(1-pp)*(1/n1+1/n2))
    return float('nan') if se==0 else 2*norm.sf(abs((p1-p2)/se))

ROOT = Path("/home/jiahao_huang/ROGUE")
recs = []
for task_name, cfg in TASK_CONFIG.items():
    ds_path = ROOT/"final_dataset"/cfg["dataset_dir"]/"all.json"
    if not ds_path.exists(): continue
    ep_idx = load_dataset(ds_path)
    reflect = {}
    for mk, fn in [("qwen3-32b","qwen3-32b_results.json"),("gpt-4.1","gpt-4.1_results.json")]:
        rp = ROOT/cfg["reflect_dir"]/fn
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
            recs.append({"avoid": av, "q1": sc["q1_ok"], "q2r": sc["q2_recall"], "q3d": sc["q3_desc_ok"]})

print(f"n={len(recs)}")
c = [r["avoid"] for r in recs if r["q1"]]
w = [r["avoid"] for r in recs if not r["q1"]]
p = z_test(len(c), sum(c)/len(c), len(w), sum(w)/len(w)) if w else float('nan')
print(f"Detection:  correct={sum(c)/len(c):.4f}(n={len(c)})  wrong={sum(w)/len(w) if w else 'nan'}(n={len(w)})  p={p:.4f}")
