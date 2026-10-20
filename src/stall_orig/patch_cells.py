"""Block 7b full-patch-token cell (STALL scoring, raw diffs, in-domain whitening). Two passes, streaming, so no token cache:
pass 1 runs the backbone on the fit clips, keeps 8 random token positions per clip (all 16 frames, so the raw token
deltas are paired), and fits STALL WhiteningTransform for tokens and raw token deltas. Pass 2 runs on fit + gt + gens:
frame log-lik = mean over all patch tokens, pair log-lik = mean over all token deltas. Aggregation {maxmin, mean};
percentiles are taken against the fit set (in-sample). Preprocessing = STALL's (squash to 224). Registers are dropped.
usage: python -m src.stall_orig.patch_cells --model M --tag vitl16 --fit DIR --real DIR --gen DIR... --out JSON"""
import argparse, glob, json, os, sys, time, numpy as np, torch
from transformers import AutoModel
from sklearn.metrics import roc_auc_score; from scipy.stats import spearmanr
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__)); from extract import TF, frames
sys.path.insert(0, os.environ.get("STALL_DIR", "/workspace/STALL") + "/src"); from whitening_transform import WhiteningTransform
FVD = {"magicdrive": 222.00, "dreamforge": 189.76, "xscene": 179.74, "drivedreamer2": 127.07, "opendwm": 90.42, "dist4d": 58.08}
ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--fit"); ap.add_argument("--real")
ap.add_argument("--gen", nargs="+"); ap.add_argument("--out"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; m = AutoModel.from_pretrained(a.model).eval().to(dev)
nreg = getattr(m.config, "num_register_tokens", 0) or 0; rng = np.random.default_rng(0)
g = lambda d: sorted(glob.glob(os.path.join(d, "*.mp4")))[: a.limit or None]
@torch.no_grad()
def toks(p): return m(pixel_values=torch.stack([TF(Image.fromarray(f)) for f in frames(p)]).to(dev)).last_hidden_state[:, 1 + nreg:].float()
t0 = time.time(); xs, ds = [], []
for p in g(a.fit):
    x = toks(p); idx = torch.as_tensor(rng.choice(x.shape[1], 8, replace=False), device=dev); x = x[:, idx]
    xs.append(x.reshape(-1, x.shape[-1]).cpu()); ds.append((x[1:] - x[:-1]).reshape(-1, x.shape[-1]).cpu())
Ws, Wt = WhiteningTransform(torch.cat(xs).to(dev)), WhiteningTransform(torch.cat(ds).to(dev)); print(f"fit done {time.time()-t0:.0f}s", flush=True)
def ll(z): return -0.5 * (z.shape[-1] * np.log(2 * np.pi) + (z ** 2).sum(-1))
def clip_ll(p):
    x = toks(p); return ll(Ws.transform(x)).mean(1).cpu().numpy(), ll(Wt.transform(x[1:] - x[:-1])).mean(1).cpu().numpy()
arms = {"fit": g(a.fit), "gt": g(a.real), **{os.path.basename(d.rstrip("/")): g(d) for d in a.gen}}
L = {k: [np.stack(v) for v in zip(*[clip_ll(p) for p in ps])] for k, ps in arms.items()}; print(f"pass2 done {time.time()-t0:.0f}s", flush=True)
AG = {"maxmin": (np.max, np.min), "mean": (np.mean, np.mean)}; out = {}
for agg, (fs, ft) in AG.items():
    cs, ct = np.sort(fs(L["fit"][0], 1)), np.sort(ft(L["fit"][1], 1))
    sc = lambda S, T: (fs(S, 1), ft(T, 1), (np.searchsorted(cs, fs(S, 1), "right") / len(cs) + np.searchsorted(ct, ft(T, 1), "right") / len(ct)) / 2)
    rs, rt, r = sc(*L["gt"]); res = {}
    for gen in [k for k in arms if k not in ("fit", "gt")]:
        gs, gt_, f = sc(*L[gen]); y = np.r_[np.ones(len(r)), np.zeros(len(f))]
        res[gen] = {"joint": roc_auc_score(y, np.r_[r, f]), "spatial": roc_auc_score(y, np.r_[rs, gs]), "temporal": roc_auc_score(y, np.r_[rt, gt_]), "mean_gen_score": float(f.mean())}
    gens = [k for k in res if k in FVD]
    rho = spearmanr([res[k]["joint"] for k in gens], [FVD[k] for k in gens])[0] if len(gens) > 2 else None
    k = f"{a.tag}|patch|nusc|{agg}|d10=False|rawdelta"
    out[k] = {"n_per_arm": len(r), "n_fit": len(L["fit"][0]), "auroc": res, "rho_auroc_vs_fvd": rho}
    print(k, " ".join(f"{q[:5]}={res[q]['joint']:.3f}" for q in res), f"rho={rho}", flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
