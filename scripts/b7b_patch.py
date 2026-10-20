"""Block 7b bridging cells built from the EXISTING primary feature cache (dinov3_s16, our resize+center-crop preprocessing,
final-norm patch tokens, registers dropped; n=150/arm, full fit_resampled 500). STALL scoring rule applied to
(a) pmean = mean-pooled patch tokens per frame, (b) patch = token-level: whiten 384-d tokens (and L2-normalised token deltas),
frame/pair log-lik = mean over 196 tokens. Whitening + calibration = in-domain fit (in-sample). agg {maxmin, mean}, d10 {F,T}."""
import glob, json, os, sys, numpy as np
sys.path.insert(0, "scripts"); from b7b_score import fitW, ll, l2, AG, GENS, FVD, OUT
from sklearn.metrics import roc_auc_score; from scipy.stats import spearmanr
RAW = os.environ.get("DELTA") == "raw"  # DELTA=raw: un-normalised deltas (as in DriveLL) instead of STALL L2-normalised
dn = (lambda d: d) if RAW else l2
S = "/workspace/data/feats/dinov3_s16"; rng = np.random.default_rng(0)
def files(arm): return sorted(glob.glob(f"{S}/{arm}__*.npy"))
def tok(f): return np.load(f).astype(np.float32)  # [16,196,384]
def lls(fs, mode, P):
    (ms, Ws), (mt, Wt) = P; S_, T_ = [], []
    for f in fs:
        x = tok(f)
        if mode == "pmean": x = x.mean(1, keepdims=True)
        d = dn(x[1:] - x[:-1])
        S_.append(ll((x - ms) @ Ws).mean(1)); T_.append(ll((d - mt) @ Wt).mean(1))
    return np.stack(S_), np.stack(T_)
def fitP(fs, mode):
    xs, ds = [], []
    for f in fs:
        x = tok(f)
        if mode == "pmean": x = x.mean(1, keepdims=True)
        else: x = x[:, rng.choice(196, 8, replace=False)]   # 8 tokens/frame -> 64k tokens for the token Gaussian
        xs.append(x.reshape(-1, 384)); ds.append(dn(x[1:] - x[:-1]).reshape(-1, 384))
    return fitW(np.concatenate(xs)), fitW(np.concatenate(ds))
def agg(S_, T_, a, d10):
    if d10: S_, T_ = np.delete(S_, 10, 1), np.delete(T_, 9, 1)
    return AG[a][0](S_, 1), AG[a][1](T_, 1)
out = {}
for mode in os.environ.get("MODES", "pmean,patch").split(","):
    P = fitP(files("fit_resampled"), mode); Fs, Ft = lls(files("fit_resampled"), mode, P)
    R = lls(files("gt"), mode, P); G = {g: lls(files(g), mode, P) for g in GENS}
    for a in os.environ.get("AGGS", "maxmin,mean").split(","):
        for d10 in [x == "True" for x in os.environ.get("D10S", "False,True").split(",")]:
            cs, ct = agg(Fs, Ft, a, d10); pc = lambda s, t: (np.searchsorted(np.sort(cs), s, "right")/len(cs) + np.searchsorted(np.sort(ct), t, "right")/len(ct))/2
            rs, rt = agg(*R, a, d10); r = pc(rs, rt); res = {}
            for g in GENS:
                gs, gt_ = agg(*G[g], a, d10); f = pc(gs, gt_); y = np.r_[np.ones(len(r)), np.zeros(len(f))]
                res[g] = {"joint": roc_auc_score(y, np.r_[r, f]), "spatial": roc_auc_score(y, np.r_[rs, gs]), "temporal": roc_auc_score(y, np.r_[rt, gt_]), "mean_gen_score": float(f.mean())}
            k = f"vits16crop|{mode}|nusc500|{a}|d10={d10}" + ("|rawdelta" if RAW else "")
            out[k] = {"n_per_arm": len(r), "auroc": res, "rho_auroc_vs_fvd": spearmanr([res[g]["joint"] for g in GENS], [FVD[g] for g in GENS])[0],
                      "rho_meanscore_vs_fvd": spearmanr([-res[g]["mean_gen_score"] for g in GENS], [FVD[g] for g in GENS])[0]}
            v = out[k]; print(k, " ".join(f"{g[:5]}={v['auroc'][g]['joint']:.3f}" for g in GENS), f"rho={v['rho_auroc_vs_fvd']:.2f} rhoM={v['rho_meanscore_vs_fvd']:.2f}", flush=True)
json.dump(out, open(os.environ.get("PATCH_OUT", f"{OUT}/patch_cache{'_rawdelta' if RAW else ''}.json"), "w"), indent=1)
