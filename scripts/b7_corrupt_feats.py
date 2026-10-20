"""Block 7.4: corruptions of the first N gt clips (post-decode, no re-encode). Caches DINOv3-S/16 patch feats + I3D feats."""
import sys, os, glob, io, numpy as np, torch
from PIL import Image, ImageFilter
sys.path.insert(0, "src"); from stall import clip_frames, BACKBONES; from features import Backbone; from baselines import i3d_feat
torch.set_num_threads(int(os.environ.get("NT", 4)))
N = int(os.environ.get("NCLIP", 50)); OUT = "/workspace/data/feats/corrupt"; os.makedirs(OUT, exist_ok=True)
SEV = {"shuffle": [2, 4, 8, 16], "dropdup": [1, 2, 4, 8], "jitter": [1, 2, 3, 4],
       "blur": [0.5, 1, 2, 4], "noise": [4, 8, 16, 32], "jpeg": [70, 50, 30, 10]}
def corrupt(f, kind, s, rng):
    f = f.copy(); T = len(f)
    if kind == "shuffle":
        pos = np.sort(rng.choice(T, s, replace=False)); perm = pos.copy()
        while (perm == pos).any(): perm = rng.permutation(pos)   # derangement of the chosen positions
        f[pos] = f[perm]
    elif kind == "dropdup":
        for t in rng.choice(np.arange(1, T), s, replace=False): f[t] = f[t - 1]
    elif kind == "jitter":
        idx = np.clip(np.arange(T) + rng.integers(-s, s + 1, T), 0, T - 1); f = f[idx]
    elif kind == "blur":
        f = np.stack([np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(s))) for x in f])
    elif kind == "noise":
        f = np.clip(f.astype(np.float32) + rng.normal(0, s, f.shape), 0, 255).astype(np.uint8)
    elif kind == "jpeg":
        def j(x):
            b = io.BytesIO(); Image.fromarray(x).save(b, "JPEG", quality=s); return np.asarray(Image.open(b).convert("RGB"))
        f = np.stack([j(x) for x in f])
    return f
paths = sorted(glob.glob("/workspace/data/enc/gt/*.mp4"))[:N]; clips = [clip_frames(p) for p in paths]
bb = Backbone(BACKBONES["dinov3_s16"])
jobs = [("clean", 0)] + [(k, s) for k, v in SEV.items() for s in v]
for k, s in jobs:
    fn = f"{OUT}/{k}_{s}"
    if os.path.exists(fn + "_i3d.npy"): continue
    P, I = [], []
    for i, f in enumerate(clips):
        g = f if k == "clean" else corrupt(f, k, s, np.random.default_rng(1000 + i))
        P.append(bb(g)[0].astype(np.float16)); I.append(i3d_feat(g))
    np.save(fn + "_dino.npy", np.stack(P)); np.save(fn + "_i3d.npy", np.stack(I)); print(k, s, flush=True)
