import math, sys
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, "/home/jiahao_huang/ROGUE")
from correlation_analysis import TASK_CONFIG, GPT41_SUFFIX, build_reflection_index, build_task_index, load_results_for_task
from score_evol_eval import compute_avoidance, load_dataset

Q2_THR = 0.6

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
            recs.append({
                "avoid": av,
                "det_ok": sc["q1_ok"],
                "loc_ok": sc["q2_recall"] >= Q2_THR,
                "diag_ok": sc["q3_desc_ok"],
            })

# Detection: all records
det_c = [r["avoid"] for r in recs if r["det_ok"]]
det_w = [r["avoid"] for r in recs if not r["det_ok"]]
p_det = z_test(len(det_c), sum(det_c)/len(det_c), len(det_w), sum(det_w)/len(det_w)) if det_w else float('nan')
print(f"Detection    n={len(recs):4d}: correct={sum(det_c)/len(det_c):.4f}(n={len(det_c)})  wrong={sum(det_w)/len(det_w) if det_w else float('nan'):.4f}(n={len(det_w)})  p={p_det:.4f}")

# Localization: only det_ok
loc_recs = [r for r in recs if r["det_ok"]]
loc_c = [r["avoid"] for r in loc_recs if r["loc_ok"]]
loc_w = [r["avoid"] for r in loc_recs if not r["loc_ok"]]
p_loc = z_test(len(loc_c), sum(loc_c)/len(loc_c), len(loc_w), sum(loc_w)/len(loc_w))
print(f"Localization n={len(loc_recs):4d}: correct={sum(loc_c)/len(loc_c):.4f}(n={len(loc_c)})  wrong={sum(loc_w)/len(loc_w):.4f}(n={len(loc_w)})  p={p_loc:.4f}")

# Diagnosis: only det_ok & loc_ok
diag_recs = [r for r in loc_recs if r["loc_ok"]]
diag_c = [r["avoid"] for r in diag_recs if r["diag_ok"]]
diag_w = [r["avoid"] for r in diag_recs if not r["diag_ok"]]
p_diag = z_test(len(diag_c), sum(diag_c)/len(diag_c), len(diag_w), sum(diag_w)/len(diag_w))
print(f"Diagnosis    n={len(diag_recs):4d}: correct={sum(diag_c)/len(diag_c):.4f}(n={len(diag_c)})  wrong={sum(diag_w)/len(diag_w):.4f}(n={len(diag_w)})  p={p_diag:.4f}")
