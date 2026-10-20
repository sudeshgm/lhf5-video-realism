"""Block 7.6: patch-level spatial/temporal Mahalanobis heatmaps for the top-3 joint-scoring clips per generator (primary protocol)."""
import sys, os, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src"); from stall import STALL, extract, clip_frames, frame_sel, delta_pairs
from features import prep, MEAN, STD
E = "/workspace/data/enc"; BB = "dinov3_s16"; cache = f"/workspace/data/feats/{BB}"; OUT = "results/figures"; os.makedirs(OUT, exist_ok=True)
g = lambda d: sorted(glob.glob(os.path.join(d, "*.mp4")))
m = STALL(drop_pos10=True).fit(extract(g("data/fit_500/fit_resampled"), BB, cache))
def maha(r, P): return ((r @ P) * r).sum(-1)
def maps(x):  # x [16,196,384] -> spatial [14,14], temporal [14,14] (mean over frames / kept deltas)
    sp = np.mean([maha(x[t].astype(np.float64) - m.gs.mu, m.gs.P) for t in frame_sel(16, True)], 0)
    tp = np.mean([maha((x[b] - x[a]).astype(np.float64) - m.gt.mu, m.gt.P) for a, b in delta_pairs(16, None, True)], 0)
    return sp.reshape(14, 14), tp.reshape(14, 14)
sp_ref, tp_ref = np.mean([maps(x)[0] for x in extract(g(f"{E}/gt"), BB, cache)[:30]]), None
out = {}
for gen in ["magicdrive", "drivedreamer2", "opendwm", "xscene", "dreamforge", "dist4d"]:
    ps = g(f"{E}/{gen}"); X = extract(ps, BB, cache); S = m.scores(X)["joint"]; top = np.argsort(-S)[:3]
    fig, ax = plt.subplots(3, 3, figsize=(11, 10))
    for r, i in enumerate(top):
        fr = clip_frames(ps[i]); img = (prep(fr[8]).transpose(1, 2, 0) * STD + MEAN).clip(0, 1)
        sp, tp = maps(X[i])
        ax[r, 0].imshow(img); ax[r, 0].set_title(f"{os.path.basename(ps[i])[:10]} joint={S[i]:.2f}\n(frame 8, 224 crop)", fontsize=8)
        for c, (mp, name) in enumerate([(sp, "spatial Mahalanobis²"), (tp, "temporal Mahalanobis² (Δ)")], 1):
            ax[r, c].imshow(img); h = ax[r, c].imshow(np.kron(mp, np.ones((16, 16))), cmap="inferno", alpha=0.6); ax[r, c].set_title(name, fontsize=8)
            plt.colorbar(h, ax=ax[r, c], fraction=0.046)
        for a in ax[r]: a.axis("off")
    fig.suptitle(f"{gen}: top-3 joint-scoring clips (DINOv3-S/16 primary protocol; per-patch mean over 15 frames / 14 deltas)", fontsize=10)
    fig.tight_layout(); fn = f"{OUT}/heatmap_top3_{gen}.png"; fig.savefig(fn, dpi=90); plt.close(fig)
    out[gen] = {"file": fn, "clips": [os.path.basename(ps[i]) for i in top], "joint": [round(float(S[i]), 3) for i in top]}; print(gen, out[gen], flush=True)
json.dump(out, open("results/block7/heatmaps.json", "w"), indent=1)
