"""Block 7b: STALL-original baseline + bridging ablation. Does NOT touch the frozen primary.
STALL recipe (OmerBenHayun/STALL@bfcc603): Gaussian log-lik of whitened frame features (spatial, agg=max over frames);
log-lik of whitened L2-normalised frame-difference vectors (temporal, agg=min over pairs); each -> percentile vs the
calibration set's aggregated scores; final = mean of percentiles. Higher = real. AUROC: real(gt)=positive.
Factors: backbone {vitl16, vits16} x feature {cls, pmean, patch} x whitening {vatex (vitl16 cls only), nusc (fit_resampled 500)}
x agg {maxmin, mean} x drop_pos10 {False (STALL exact), True (our amended protocol)}."""
import glob, json, os, sys, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
SD = os.environ.get("STALL_DIR", "/workspace/STALL"); sys.path.insert(0, SD + "/src"); from whitening_transform import WhiteningTransform
import torch
FVD = {"magicdrive": 222.00, "dreamforge": 189.76, "xscene": 179.74, "drivedreamer2": 127.07, "opendwm": 90.42, "dist4d": 58.08}
GENS = list(FVD); C = os.environ.get("STALL_CACHE", "/workspace/data/feats/stall_orig"); OUT = "results/block7b"
VATEX = np.load(SD + "/precomputed/stall_params_vatex_dino_v3.npz")
def ll(z): return -0.5 * (z.shape[-1] * np.log(2 * np.pi) + (z ** 2).sum(-1))
def l2(x): n = np.linalg.norm(x, axis=-1, keepdims=True); return x / np.where(n == 0, 1, n)
def fitW(X):
    wt = WhiteningTransform(torch.tensor(np.ascontiguousarray(X), dtype=torch.float32)); return wt.mean_.numpy(), wt.whitening_matrix_.numpy()
def load(tag, arm, feat, src=None):
    fs = sorted(glob.glob(f"{C}/{tag}/{arm}__*.npz"))
    if src is not None: fs = [f for f in fs if os.path.basename(f)[len(arm)+2:-4] in src]
    return [os.path.basename(f)[len(arm)+2:-4] for f in fs], np.stack([np.load(f)[feat] for f in fs])
def drop10(E): return np.delete(E, 10, axis=1)
def tdiff(E, d10):  # L2-normalised diffs; with d10 drop the 9->10 delta (14 deltas) like the primary amendment
    D = l2(E[:, 1:] - E[:, :-1]); return np.delete(D, 9, axis=1) if d10 else D
AG = {"maxmin": (np.max, np.min), "mean": (np.mean, np.mean)}
def score(params, calib, E, agg, d10):
    (ms, Ws), (mt, Wt) = params; fs, ft = AG[agg]
    Es = drop10(E) if d10 else E
    s = fs(ll((Es - ms) @ Ws), 1); t = ft(ll((tdiff(E, d10) - mt) @ Wt), 1)
    cs, ct = calib; ps = np.searchsorted(np.sort(cs), s, "right") / len(cs); pt = np.searchsorted(np.sort(ct), t, "right") / len(ct)
    return (ps + pt) / 2, s, t
def run_cell(tag, feat, wh, agg, d10, sub=None):
    names, R = load(tag, "gt", feat, sub)
    if wh == "vatex":
        params = ((VATEX["mu_spat"], VATEX["W_spat"]), (VATEX["mu_temp"], VATEX["W_temp"]))
        fs, ft = AG[agg]; calib = (fs(VATEX["calib_ll_spat"], 1), ft(VATEX["calib_ll_temp"], 1))
    else:
        _, F = load(tag, "fit_resampled", feat)
        Fs = drop10(F) if d10 else F; Ft = tdiff(F, d10)
        params = (fitW(Fs.reshape(-1, F.shape[-1])), fitW(Ft.reshape(-1, F.shape[-1])))
        _, cs, ct = score(params, (np.zeros(1), np.zeros(1)), F, agg, d10); calib = (cs, ct)  # in-sample calibration (fit set)
    r, rs, rt = score(params, calib, R, agg, d10); res = {}
    for g in GENS:
        gn, G = load(tag, g, feat, sub); assert gn == names
        f, fs_, ft_ = score(params, calib, G, agg, d10); y = np.r_[np.ones(len(r)), np.zeros(len(f))]
        res[g] = {"joint": roc_auc_score(y, np.r_[r, f]), "spatial": roc_auc_score(y, np.r_[rs, fs_]), "temporal": roc_auc_score(y, np.r_[rt, ft_]), "mean_gen_score": float(f.mean())}
    rho = spearmanr([res[g]["joint"] for g in GENS], [FVD[g] for g in GENS])[0]
    rho_mean = spearmanr([-res[g]["mean_gen_score"] for g in GENS], [FVD[g] for g in GENS])[0]
    return {"n_per_arm": len(r), "auroc": res, "rho_auroc_vs_fvd": rho, "rho_meanscore_vs_fvd": rho_mean}
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); which = sys.argv[1]; out = {}
    if which == "vitl":  # STALL-original on the feasible ViT-L subset (10 paired clips/arm; CPU-feasible smoke), VATEX whitening only
        sub = set(os.path.basename(p) for p in glob.glob("/workspace/data/sub10/gt/*.mp4"))
        for agg in AG:
            for d10 in (False, True): out[f"vitl16|cls|vatex|{agg}|d10={d10}"] = run_cell("vitl16", "cls", "vatex", agg, d10, sub)
            out[f"vitl16|pmean|nusc|{agg}|d10=True"] = None  # needs fit_500 ViT-L features -> GPU job
    elif which == "vitl_full":  # GPU job (scripts/gpu/stall_orig.sh): all 150/arm, VATEX + in-domain whitening
        for feat in ("cls", "pmean"):
            for wh in (("vatex", "nusc") if feat == "cls" else ("nusc",)):
                for agg in AG:
                    for d10 in (False, True): out[f"vitl16|{feat}|{wh}|{agg}|d10={d10}"] = run_cell("vitl16", feat, wh, agg, d10)
    else:
        for feat in ("cls", "pmean"):
            for agg in AG:
                for d10 in (False, True): out[f"vits16|{feat}|nusc|{agg}|d10={d10}"] = run_cell("vits16", feat, "nusc", agg, d10)
    out = {k: v for k, v in out.items() if v is not None}
    json.dump(out, open(f"{OUT}/{which}.json", "w"), indent=1)
    for k, v in out.items():
        print(k, "n=%d" % v["n_per_arm"], " ".join(f"{g[:5]}={v['auroc'][g]['joint']:.3f}" for g in GENS), f"rho={v['rho_auroc_vs_fvd']:.2f} rhoM={v['rho_meanscore_vs_fvd']:.2f}")
