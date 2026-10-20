"""Training-free STALL-style likelihood scoring on frozen ViT patch tokens.
Fit a Ledoit-Wolf Gaussian on REAL fit clips only; score = mean squared Mahalanobis distance.
  spatial : patch tokens x[t,n]            (T*N tokens per clip)
  temporal: x[t+1,n]-x[t,n]                ((T-1)*N tokens per clip)
  joint   : z_spatial + z_temporal, z-stats (mean/std) from fit-set scores only.
Higher score = less real-like. AUROC: gen = positive class, 1000x stratified bootstrap CI.
"""
import argparse, glob, json, os, sys, time
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

BACKBONES = {"dinov3_s16": "facebook/dinov3-vits16-pretrain-lvd1689m",  # primary (pre-registered)
             "dinov2_b14": "facebook/dinov2-base",                       # pre-registered rerun
             "dinov2_s14": "facebook/dinov2-small"}                      # dry-run / plumbing only

class Gauss:
    """Ledoit-Wolf Gaussian over token vectors. per_position: mean per patch index n, shared covariance."""
    def __init__(self, per_position=False, max_tokens=60000, seed=0):
        self.pp, self.max, self.rng = per_position, max_tokens, np.random.default_rng(seed)
    def fit(self, clips):  # list of [T,N,D]
        X = np.stack(clips).astype(np.float64)               # [C,T,N,D]
        self.mu = X.mean((0, 1)) if self.pp else X.reshape(-1, X.shape[-1]).mean(0)  # [N,D] or [D]
        R = (X - self.mu).reshape(-1, X.shape[-1])
        if len(R) > self.max: R = R[self.rng.choice(len(R), self.max, replace=False)]
        lw = LedoitWolf(assume_centered=True).fit(R); self.shrinkage = float(lw.shrinkage_)
        self.P = lw.precision_; return self
    def score(self, x):  # [T,N,D] -> mean squared Mahalanobis
        r = (x.astype(np.float64) - self.mu).reshape(-1, x.shape[-1])
        return float(((r @ self.P) * r).sum(1).mean())

def tdiff(x): return x[1:] - x[:-1]

class STALL:
    def __init__(self, per_position=False):
        self.gs, self.gt = Gauss(per_position), Gauss(per_position)
    def fit(self, clips):
        self.gs.fit(clips); self.gt.fit([tdiff(c) for c in clips])
        S = self.raw(clips); self.z = {k: (S[k].mean(), S[k].std() + 1e-12) for k in ("spatial", "temporal")}; return self
    def raw(self, clips):
        return {"spatial": np.array([self.gs.score(c) for c in clips]),
                "temporal": np.array([self.gt.score(tdiff(c)) for c in clips])}
    def scores(self, clips):
        S = self.raw(clips); Z = {k: (S[k] - self.z[k][0]) / self.z[k][1] for k in S}
        S["joint"] = Z["spatial"] + Z["temporal"]; return S

def auroc_ci(real, gen, n_boot=1000, seed=0):
    y = np.r_[np.zeros(len(real)), np.ones(len(gen))]; s = np.r_[real, gen]
    a = roc_auc_score(y, s); rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        r = rng.choice(real, len(real)); g = rng.choice(gen, len(gen))
        bs.append(roc_auc_score(y, np.r_[r, g]))
    lo, hi = np.percentile(bs, [2.5, 97.5]); return float(a), float(lo), float(hi)

def verdict(a):  # pre-registered stop rule
    return "alive" if a >= 0.70 else "weak" if a >= 0.60 else "near-chance"

# ---------------- feature extraction (cached) ----------------
def clip_frames(path):
    import av
    fr = [f.to_ndarray(format="rgb24") for f in av.open(path).decode(video=0)]
    assert len(fr) == 16, (path, len(fr)); return np.stack(fr)

def extract(paths, backbone, cache_dir):
    sys.path.insert(0, os.path.dirname(__file__)); from features import Backbone
    os.makedirs(cache_dir, exist_ok=True); bb = None; out = []
    for p in paths:
        c = os.path.join(cache_dir, os.path.basename(os.path.dirname(p)) + "__" + os.path.basename(p) + ".npy")
        if not os.path.exists(c):
            bb = bb or Backbone(BACKBONES[backbone]); np.save(c, bb(clip_frames(p))[0].astype(np.float16))
        out.append(np.load(c).astype(np.float32))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--fit", required=True); ap.add_argument("--real", required=True)
    ap.add_argument("--gen", required=True, nargs="+")
    ap.add_argument("--per-position", action="store_true")
    ap.add_argument("--cache", default="/workspace/data/feats"); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(); t0 = time.time(); g = lambda d: sorted(glob.glob(os.path.join(d, "*.mp4")))
    cache = os.path.join(a.cache, a.backbone)
    F, R = extract(g(a.fit), a.backbone, cache), extract(g(a.real), a.backbone, cache)
    m = STALL(a.per_position).fit(F); sr = m.scores(R)
    res = {"label": a.label, "backbone": a.backbone, "model_id": BACKBONES[a.backbone], "per_position": a.per_position,
           "n_fit": len(F), "n_real": len(R), "feat_shape": list(F[0].shape),
           "lw_shrinkage": {"spatial": m.gs.shrinkage, "temporal": m.gt.shrinkage}, "gen": {}}
    for gd in a.gen:
        G = extract(g(gd), a.backbone, cache); sg = m.scores(G); r = {"n_gen": len(G)}
        for k in ("spatial", "temporal", "joint"):
            au, lo, hi = auroc_ci(sr[k], sg[k]); r[k] = {"auroc": round(au, 4), "ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict(au)}
        res["gen"][os.path.basename(gd.rstrip("/"))] = r
    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))

if __name__ == "__main__": main()
