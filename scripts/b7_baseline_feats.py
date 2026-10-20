"""Cache I3D (per clip) and Inception (per frame) features for gt + 6 gen0 (+ optional extra dirs)."""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "src"); from baselines import i3d_feat, inception_feats; from stall import clip_frames
torch.set_num_threads(int(os.environ.get("NT", 4)))
E = "/workspace/data/enc"; C = "/workspace/data/feats"
INC_FRAMES = [0, 4, 8, 12]  # FID: 4 frames/clip (CPU budget)
arms = sys.argv[1:] or ["gt", "magicdrive", "drivedreamer2", "opendwm", "xscene", "dreamforge", "dist4d"]
for a in arms:
    if os.path.exists(f"{C}/inception/{a}.npy"): continue
    os.makedirs(f"{C}/i3d", exist_ok=True); os.makedirs(f"{C}/inception", exist_ok=True)
    fs = sorted(glob.glob(f"{E}/{a}/*.mp4")); I, F = [], []
    for p in fs:
        fr = clip_frames(p); F.append(inception_feats(fr[INC_FRAMES]))
    np.save(f"{C}/inception/{a}.npy", np.stack(F)); print(a, len(fs), flush=True)
