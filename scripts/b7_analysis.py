"""Block 7 analysis: baselines (1), sample efficiency (2), fit-set efficiency (3), corruption monotonicity (4).
Primary protocol frozen (DINOv3-S/16, drop-pos10, joint z-sum, no dedupe)."""
import sys, os, glob, json, numpy as np
from scipy.stats import spearmanr, kendalltau
sys.path.insert(0, "src"); from stall import STALL, extract, auroc_ci; from baselines import frechet, dino_clip_vec, FrechetRef
E = "/workspace/data/enc"; C = "/workspace/data/feats"; BB = "dinov3_s16"; OUT = "results/block7"; os.makedirs(OUT, exist_ok=True)
GENS = ["magicdrive", "drivedreamer2", "opendwm", "xscene", "dreamforge", "dist4d"]
WL = {"magicdrive": 222.00, "dreamforge": 189.76, "drivedreamer2": 127.07, "opendwm": 90.42, "dist4d": 58.08, "xscene": 179.74}
g = lambda d: sorted(glob.glob(os.path.join(d, "*.mp4")))
rho = lambda a, b: float(spearmanr(a, b).statistic)
parts = sys.argv[1:] or ["1", "2", "3", "4"]
CS = json.load(open("results/robustness/clip_scores.json"))
ours_mean = {k: float(np.mean(CS[k]["joint"])) for k in GENS}
res = json.load(open(f"{OUT}/analysis.json")) if os.path.exists(f"{OUT}/analysis.json") else {}
def save(): json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=1)
if "1" in parts or "2" in parts:
    I3 = {a: np.load(f"{C}/i3d/{a}.npy").astype(np.float64) for a in ["gt"] + GENS}
    HAVE_INC = all(os.path.exists(f"{C}/inception/{a}.npy") for a in ["gt"] + GENS)
    INC = {} if ("1" not in parts or not HAVE_INC) else {a: (lambda z: (z[:, [0, 4, 8, 12]] if z.shape[1] == 16 else z))(np.load(f"{C}/inception/{a}.npy")).reshape(-1, 2048).astype(np.float64) for a in ["gt"] + GENS}
    DV = {a: np.stack([dino_clip_vec(x) for x in extract(g(f"{E}/{a}"), BB, f"{C}/{BB}")]).astype(np.float64) for a in ["gt"] + GENS}
if "1" in parts:
    B = {"FVD_I3D": {k: frechet(I3["gt"], I3[k]) for k in GENS},
         **({"FID_inception_perframe": {k: frechet(INC["gt"], INC[k]) for k in GENS}} if INC else {}),
         "FD_DINOv3": {k: frechet(DV["gt"], DV[k]) for k in GENS}, "ours_mean_clip_joint": ours_mean}
    real_split = {"FVD_I3D": frechet(I3["gt"][::2], I3["gt"][1::2]), "FD_DINOv3": frechet(DV["gt"][::2], DV["gt"][1::2])}
    if INC: real_split["FID_inception_perframe"] = frechet(INC["gt"][: 75 * 4], INC["gt"][75 * 4:])
    sp = {m: {"vs_WorldLens_FVD": rho([v[k] for k in GENS], [WL[k] for k in GENS]), "vs_ours": rho([v[k] for k in GENS], [ours_mean[k] for k in GENS])} for m, v in B.items()}
    res["1_baselines"] = {"values": B, "gt_half_vs_half_floor (75 vs 75)": real_split, "spearman_n6": sp,
        "notes": "FVD-I3D: StyleGAN-V i3d_torchscript (400-d), 150 gt vs 150 gen clips, 16 frames @456x256 resized to 224. FID: pytorch-fid InceptionV3 pool3, frames 0,4,8,12 per clip (600 vs 600). FD-DINOv3: clip vec = [mean patch token over 15 frames (pos10 dropped); mean |delta| over 14 deltas], 768-d, n=150 (cov rank-deficient)."}
    save(); print(json.dumps(res["1_baselines"]["spearman_n6"], indent=0), flush=True)
if "2" in parts:
    full = {"ours": [ours_mean[k] for k in GENS], "FVD_I3D": [frechet(I3["gt"], I3[k]) for k in GENS], "FD_DINOv3": [frechet(DV["gt"], DV[k]) for k in GENS]}
    out = {}
    for kk in [5, 10, 25, 50, 150]:
        FI, FD = FrechetRef(I3["gt"]), FrechetRef(DV["gt"])
        rs = {m: [] for m in full}
        for b in range(200):
            rng = np.random.default_rng(b); vals = {m: [] for m in full}
            for k in GENS:
                idx = rng.choice(150, kk, replace=(kk == 150))
                vals["ours"].append(float(np.mean(np.array(CS[k]["joint"])[idx])))
                vals["FVD_I3D"].append(FI(I3[k][idx])); vals["FD_DINOv3"].append(FD(DV[k][idx]))
            for m in full: rs[m].append(rho(vals[m], full[m]))
        out[kk] = {m: {"median": round(float(np.nanmedian(v)), 3), "iqr": [round(float(np.nanpercentile(v, 25)), 3), round(float(np.nanpercentile(v, 75)), 3)], "mean": round(float(np.nanmean(v)), 3), "sd": round(float(np.nanstd(v)), 3), "p_rho_eq_1": round(float(np.mean(np.isclose(v, 1))), 3)} for m, v in rs.items()}
        print(kk, out[kk], flush=True)
    res["2_sample_efficiency"] = {"protocol": "k gen clips per generator (k=150: bootstrap w/ replacement), 200 resamples; real reference = all 150 gt for FVD/FD; rho vs the same metric's full-n ranking over 6 generators", "by_k": out}
    save()
if "3" in parts:
    F = extract(g("data/fit_500/fit_resampled"), BB, f"{C}/{BB}"); R = extract(g(f"{E}/gt"), BB, f"{C}/{BB}")
    G = {k: extract(g(f"{E}/{k}"), BB, f"{C}/{BB}") for k in GENS}; out = {}
    for seed in [0, 1, 2]:
        perm = np.random.default_rng(seed).permutation(len(F))
        for n in [50, 100, 250, 500]:
            if n == 500 and seed > 0: continue
            m = STALL(drop_pos10=True).fit([F[i] for i in perm[:n]]); sr = m.scores(R)["joint"]; au = {}; mj = {}
            for k in GENS:
                sg = m.scores(G[k])["joint"]; au[k] = round(auroc_ci(sr, sg, n_boot=200)[0], 4); mj[k] = float(sg.mean())
            out[f"seed{seed}_n{n}"] = {"auroc": au, "rho_meanjoint_vs_WorldLens": round(rho([mj[k] for k in GENS], [WL[k] for k in GENS]), 3),
                                       "rho_vs_full_fit_ranking": round(rho([mj[k] for k in GENS], [ours_mean[k] for k in GENS]), 3)}
            print(seed, n, out[f"seed{seed}_n{n}"], flush=True)
    res["3_fit_efficiency"] = {"protocol": "nested subsets of fit_resampled (permutation per seed); AUROC point estimates (200-boot for speed; CI not reported)", "runs": out}; save()
if "4" in parts:
    F = extract(g("data/fit_500/fit_resampled"), BB, f"{C}/{BB}"); m = STALL(drop_pos10=True).fit(F); del F
    I3gt = np.load(f"{C}/i3d/gt.npy").astype(np.float64); ref_i3d = I3gt[50:]  # disjoint reference (gt clips 50..149)
    SEV = {"shuffle": [2, 4, 8, 16], "dropdup": [1, 2, 4, 8], "jitter": [1, 2, 3, 4], "blur": [0.5, 1, 2, 4], "noise": [4, 8, 16, 32], "jpeg": [70, 50, 30, 10]}
    D = f"{C}/corrupt"; clean = list(np.load(f"{D}/clean_0_dino.npy").astype(np.float32)); sc0 = m.scores(clean); f0 = frechet(ref_i3d, np.load(f"{D}/clean_0_i3d.npy").astype(np.float64))
    out = {}
    for kind, sv in SEV.items():
        rows = [{"sev": 0, **{k: float(np.mean(sc0[k])) for k in sc0}, "FVD_I3D_vs_gt100": f0}]
        for s in sv:
            X = list(np.load(f"{D}/{kind}_{s}_dino.npy").astype(np.float32)); sc = m.scores(X)
            rows.append({"sev": s, **{k: float(np.mean(sc[k])) for k in sc}, "FVD_I3D_vs_gt100": frechet(ref_i3d, np.load(f"{D}/{kind}_{s}_i3d.npy").astype(np.float64))})
        lv = list(range(5))
        out[kind] = {"rows": rows, **{f"kendall_{k}": round(float(kendalltau(lv, [r[k] for r in rows]).statistic), 3) for k in ["spatial", "temporal", "joint", "FVD_I3D_vs_gt100"]}}
        print(kind, {k: v for k, v in out[kind].items() if k != "rows"}, flush=True)
    res["4_corruptions"] = {"protocol": "first 50 gt clips (sorted), corruption applied post-decode (no re-encode), seed 1000+i; scores = mean over clips; FVD = I3D Frechet of the 50 corrupted vs disjoint gt clips 50..149; Kendall tau over severities 0..4", "by_corruption": out}; save()
