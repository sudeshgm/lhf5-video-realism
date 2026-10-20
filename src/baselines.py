"""Distribution-level baselines on the SAME re-encoded clips.
- FVD-I3D: StyleGAN-V I3D torchscript (Kinetics-400, 400-d logits features, rescale=False, resize=True; input [-1,1]).
- FID (per-frame): pytorch-fid InceptionV3 pool3 (2048-d), all 16 frames per clip.
- DINOv3-FD: Frechet distance on clip vectors [mean patch token (15 frames, pos10 dropped) ; mean |delta| (14 deltas)] = 2D dims."""
import os, numpy as np, torch
from scipy import linalg
I3D_PATH = "/workspace/data/models/i3d_torchscript.pt"
_i3d = _inc = None

def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = np.real(linalg.sqrtm(s1 @ s2))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))

@torch.no_grad()
def i3d_feat(frames):  # [T,H,W,3] uint8 -> [400]
    global _i3d
    if _i3d is None: _i3d = torch.jit.load(I3D_PATH).eval()
    x = (torch.from_numpy(frames).float().permute(3, 0, 1, 2)[None] / 127.5 - 1).contiguous()
    return _i3d(x, rescale=False, resize=True, return_features=True)[0].numpy()

@torch.no_grad()
def inception_feats(frames):  # [T,H,W,3] -> [T,2048]
    global _inc
    if _inc is None:
        from pytorch_fid.inception import InceptionV3
        _inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    return _inc(x)[0][:, :, 0, 0].numpy()

def dino_clip_vec(x, pos=10):  # x [T,N,D] patch tokens
    keep = [t for t in range(len(x)) if t != pos]
    d = np.stack([x[b] - x[a] for a, b in zip(range(len(x) - 1), range(1, len(x))) if a != pos - 1])
    return np.r_[x[keep].mean((0, 1)), np.abs(d).mean((0, 1))]

class FrechetRef:
    """Fast exact Frechet distance to a FIXED reference set: tr sqrt(S1 S2) = sum sqrt(eig((Y A)(Y A)^T/(k-1))),
    A = S1^{1/2}, Y = centered samples (k x d). Equivalent to frechet(ref, X)."""
    def __init__(self, ref):
        self.mu = ref.mean(0); S = np.cov(ref, rowvar=False); w, V = np.linalg.eigh(S)
        self.A = (V * np.sqrt(np.clip(w, 0, None))) @ V.T; self.trS = float(np.trace(S))
    def __call__(self, X):
        mu = X.mean(0); Y = X - mu; k = len(X); B = Y @ self.A
        ev = np.linalg.eigvalsh(B @ B.T / (k - 1)); trS2 = float((Y * Y).sum() / (k - 1))
        return float(((self.mu - mu) ** 2).sum() + self.trS + trS2 - 2 * np.sqrt(np.clip(ev, 0, None)).sum())
