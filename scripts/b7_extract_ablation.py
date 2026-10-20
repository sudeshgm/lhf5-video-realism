"""Block 7.5: cache features for backbone ablation: gt + 6 gen0, then fit_resampled.
FITN (env, default 500): only the first FITN clips of the seed-0 permutation of fit_resampled (CPU budget)."""
import sys, os, glob, time, numpy as np
sys.path.insert(0, "src"); from stall import extract
import torch; torch.set_num_threads(int(os.environ.get("NT", 4)))
E = "/workspace/data/enc"; FITN = int(os.environ.get("FITN", 500))
fit = sorted(glob.glob("data/fit_500/fit_resampled/*.mp4")); fit = [fit[i] for i in np.random.default_rng(0).permutation(len(fit))[:FITN]]
for bb in sys.argv[1:]:
    t = time.time()
    for d in ["gt", "magicdrive", "drivedreamer2", "opendwm", "xscene", "dreamforge", "dist4d"]:
        extract(sorted(glob.glob(f"{E}/{d}/*.mp4")), bb, f"/workspace/data/feats/{bb}", load=False); print(bb, d, flush=True)
    extract(fit, bb, f"/workspace/data/feats/{bb}", load=False)
    print(bb, "done", round(time.time() - t), "s", flush=True)
