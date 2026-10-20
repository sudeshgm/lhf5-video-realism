"""Block 6 robustness pass. SAME frozen primary protocol: DINOv3-S/16, fit_resampled (500), no dedupe,
drop-pos10, joint = z_spatial + z_temporal (fit-set z-stats), 1000x bootstrap. No metric changes."""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from stall import STALL, auroc_ci, verdict, extract, consec_mae, n_repeats, delta_pairs
E = "/workspace/data/enc"; FIT = "data/fit_500/fit_resampled"; BB = "dinov3_s16"
cache = f"/workspace/data/feats/{BB}"; mc = "/workspace/data/feats/mae"; out = "results/robustness"; os.makedirs(out, exist_ok=True)
g = lambda d: sorted(glob.glob(os.path.join(d, "*.mp4")))
ARMS = ["magicdrive", "drivedreamer2", "opendwm", "xscene", "dreamforge", "dist4d"] + \
       [f"{m}_gen{k}" for m in ("magicdrive", "drivedreamer2") for k in (1, 2, 3)]
F = extract(g(FIT), BB, cache); R = extract(g(f"{E}/gt"), BB, cache)
m = STALL(per_position=False, drop_pos10=True).fit(F); sr = m.scores(R)
nrep = lambda d: -np.array([n_repeats(consec_mae(p, mc), True) for p in g(d)], float)
rr = nrep(f"{E}/gt")
res = {"protocol": "PRIMARY frozen: dinov3_s16, fit_resampled n=500, no dedupe, drop_pos10, joint z-sum, 1000x bootstrap",
       "n_fit": len(F), "n_real": len(R), "real_mean_joint": float(sr["joint"].mean()), "arms": {}}
clip = {"gt": {k: v.tolist() for k, v in sr.items()}, "gt_names": [os.path.basename(p) for p in g(f"{E}/gt")]}
def per_delta(clips):  # [n_clips, 14] mean sq. Mahalanobis per kept delta position
    P = delta_pairs(16, None, True); o = []
    for c in clips:
        r = np.stack([c[b] - c[a] for a, b in P]).astype(np.float64) - m.gt.mu
        o.append([float(((x @ m.gt.P) * x).sum(1).mean()) for x in r])
    return np.array(o), P
for a in ARMS:
    G = extract(g(f"{E}/{a}"), BB, cache); assert [os.path.basename(p) for p in g(f"{E}/{a}")] == clip["gt_names"]
    sg = m.scores(G); r = {"n_gen": len(G), "mean_joint": float(sg["joint"].mean()), "median_joint": float(np.median(sg["joint"]))}
    for k in ("spatial", "temporal", "joint"):
        au, lo, hi = auroc_ci(sr[k], sg[k]); r[k] = {"auroc": round(au, 4), "ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict(au)}
    au, lo, hi = auroc_ci(rr, nrep(f"{E}/{a}")); r["shortcut_nrepeats_excl9to10"] = {"auroc": round(au, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    r["FLAG_joint_not_above_shortcut"] = bool(r["joint"]["auroc"] <= au)
    res["arms"][a] = r; clip[a] = {k: v.tolist() for k, v in sg.items()}
    if a == "magicdrive":
        dr, P = per_delta(R); dg, _ = per_delta(G)
        res["magicdrive_per_delta"] = [{"pair": [int(x), int(y)], **dict(zip(("auroc", "lo", "hi"), [round(v, 4) for v in auroc_ci(dr[:, i], dg[:, i])]))} for i, (x, y) in enumerate(P)]
    print(a, {k: r[k]["auroc"] for k in ("spatial", "temporal", "joint")}, "shortcut", r["shortcut_nrepeats_excl9to10"]["auroc"], flush=True)
seeds = {}
for mname in ("magicdrive", "drivedreamer2"):
    v = [res["arms"][mname]["joint"]["auroc"]] + [res["arms"][f"{mname}_gen{k}"]["joint"]["auroc"] for k in (1, 2, 3)]
    seeds[mname] = {"gen0..3": v, "mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)}
res["seeds"] = seeds
json.dump(res, open(f"{out}/robustness.json", "w"), indent=1); json.dump(clip, open(f"{out}/clip_scores.json", "w"))
print(json.dumps(seeds), json.dumps(res["magicdrive_per_delta"]))
