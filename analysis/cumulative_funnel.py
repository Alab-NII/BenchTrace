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

for cat in ["operation", "strategy"]:
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
                if r["target_failure_instance"]["type"].split("/")[0] != cat: continue
                recs.append({
                    "avoid":   av,
                    "det_ok":  sc["q1_ok"],
                    "loc_ok":  sc["q2_recall"] >= Q2_THR,
                    "diag_ok": sc["q3_desc_ok"],
                })

    N = len(recs)
    print(f"\n=== {cat.upper()} (N={N}) ===")
    print(f"{'Level':<8} {'n':>5} {'FAR|correct':>12} {'FAR|wrong':>11} {'p':>9}")
    print("-" * 50)

    levels = [
        ("Det.",     lambda r: r["det_ok"]),
        ("Loc.",     lambda r: r["det_ok"] and r["loc_ok"]),
        ("Diag.",    lambda r: r["det_ok"] and r["loc_ok"] and r["diag_ok"]),
    ]
    for label, crit in levels:
        c = [r["avoid"] for r in recs if crit(r)]
        w = [r["avoid"] for r in recs if not crit(r)]
        if not c or not w:
            print(f"  {label:<6} {N:>5} {len(c):>12}  —")
            continue
        p = z_test(len(c), sum(c)/len(c), len(w), sum(w)/len(w))
        print(f"  {label:<6} {N:>5} {sum(c)/len(c):>12.4f}(n={len(c)})  {sum(w)/len(w):>7.4f}(n={len(w)})  {p:>9.4f}")
