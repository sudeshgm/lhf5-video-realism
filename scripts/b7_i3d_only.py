"""I3D-only feature cache (priority for sample-efficiency item), batched; Inception is filled later by b7_baseline_feats.py."""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "src"); from stall import clip_frames
torch.set_num_threads(int(os.environ.get("NT", 6))); C = "/workspace/data/feats/i3d"
m = torch.jit.load("/workspace/data/models/i3d_torchscript.pt").eval()
for a in sys.argv[1:]:
    if os.path.exists(f"{C}/{a}.npy"): continue
    ps = sorted(glob.glob(f"/workspace/data/enc/{a}/*.mp4")); out = []
    for i in range(0, len(ps), 10):
        x = torch.from_numpy(np.stack([clip_frames(p) for p in ps[i:i + 10]])).float().permute(0, 4, 1, 2, 3) / 127.5 - 1
        with torch.no_grad(): out.append(m(x.contiguous(), rescale=False, resize=True, return_features=True).numpy())
        print(a, i, flush=True)
    np.save(f"{C}/{a}.npy", np.concatenate(out)); print(a, "saved", flush=True)
