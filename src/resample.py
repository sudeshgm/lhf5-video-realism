"""Shared frame resampler (used by scripts/azure/export_fit.py and scripts/build_mini_fit.py).
Nearest-timestamp sampling of a raw CAM_FRONT chain (samples+sweeps) onto a uniform 1/hz grid
starting at the first frame given (caller passes the chain starting at the first keyframe)."""
import numpy as np

def resample_indices(timestamps_us, T=16, hz=12.0):
    ts = np.asarray(timestamps_us, dtype=np.float64)
    if len(ts) < 2: raise ValueError("chain too short")
    grid = ts[0] + np.arange(T) * 1e6 / hz
    if grid[-1] > ts[-1] + 0.5e6 / hz:
        raise ValueError(f"chain too short: need {grid[-1]-ts[0]:.0f}us, have {ts[-1]-ts[0]:.0f}us")
    idx = np.searchsorted(ts, grid)
    idx = np.clip(idx, 1, len(ts) - 1)
    left, right = ts[idx - 1], ts[idx]
    idx = np.where(grid - left <= right - grid, idx - 1, idx)  # ties -> earlier frame
    return idx.astype(int).tolist()

def raw_indices(n_available, T=16):
    if n_available < T: raise ValueError("chain too short")
    return list(range(T))
