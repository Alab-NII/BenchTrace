import json, math, sys
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, "/home/jiahao_huang/ROGUE")
from correlation_analysis import (
    TASK_CONFIG, GPT41_SUFFIX, Q3_DESC_THRESHOLD,
    build_reflection_index, build_task_index, load_results_for_task,
)
from score_evol_eval import compute_avoidance, load_dataset

Q2_THR = 0.6

def z_test(n1, p1, n2, p2):
    p_pool = (p1*n1 + p2*n2) / (n1 + n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    if se == 0:
        return float('nan')
    return 2 * norm.sf(abs((p1 - p2) / se))

all_records = []
ROOT = Path("/home/jiahao_huang/ROGUE")

for task_name, cfg in TASK_CONFIG.items():
    ds_path = ROOT / "final_dataset" / cfg["dataset_dir"] / "all.json"
    if not ds_path.exists():
        continue
    episode_index = load_dataset(ds_path)
    reflect = {}
    for model_key, fname in [("qwen3-32b", "qwen3-32b_results.json"),
                              ("gpt-4.1",   "gpt-4.1_results.json")]:
        rpath = ROOT / cfg["reflect_dir"] / fname
        if rpath.exists():
            reflect[model_key] = build_reflection_index(rpath)
    task_index = build_task_index(task_name, cfg)
    by_baseline = load_results_for_task(cfg)
    for baseline, results in by_baseline.items():
        model = "gpt-4.1" if baseline.endswith(GPT41_SUFFIX) else "qwen3-32b"
        refl  = reflect.get(model, {})
        for r in results:
            tid = r.get("task_id", "")
            signal_id = task_index.get(tid)
            if signal_id is None:
                continue
            scores = refl.get(signal_id)
            if scores is None:
                continue
            avoid = compute_avoidance(r, episode_index, cfg["use_location"], task_name)
            if avoid is None:
                continue
            cat = r["target_failure_instance"]["type"].split("/")[0]
            if cat != "strategy":
                continue
            q2_ok = scores["q2_recall"] >= Q2_THR
            all_records.append({
                "avoid":   avoid,
                "loc_ok":  q2_ok,
                "diag_ok": scores["q3_desc_ok"],
            })

print(f"Strategy records: {len(all_records)}")

loc_c = [r["avoid"] for r in all_records if r["loc_ok"]]
loc_w = [r["avoid"] for r in all_records if not r["loc_ok"]]
p_loc = z_test(len(loc_c), sum(loc_c)/len(loc_c), len(loc_w), sum(loc_w)/len(loc_w))
print(f"Localization: correct={sum(loc_c)/len(loc_c):.4f} (n={len(loc_c)})  "
      f"wrong={sum(loc_w)/len(loc_w):.4f} (n={len(loc_w)})  p={p_loc:.4f}")

loc_ok_recs = [r for r in all_records if r["loc_ok"]]
diag_c = [r["avoid"] for r in loc_ok_recs if r["diag_ok"]]
diag_w = [r["avoid"] for r in loc_ok_recs if not r["diag_ok"]]
if diag_c and diag_w:
    p_diag = z_test(len(diag_c), sum(diag_c)/len(diag_c),
                    len(diag_w), sum(diag_w)/len(diag_w))
    print(f"Diagnosis:    correct={sum(diag_c)/len(diag_c):.4f} (n={len(diag_c)})  "
          f"wrong={sum(diag_w)/len(diag_w):.4f} (n={len(diag_w)})  p={p_diag:.4f}")
else:
    print(f"Diagnosis: diag_c={len(diag_c)}  diag_w={len(diag_w)}")
