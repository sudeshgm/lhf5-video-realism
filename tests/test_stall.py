import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stall import STALL, auroc_ci
T, N, D = 16, 16, 32
def clips(n, shift, seed, temporal=False):
    r = np.random.default_rng(seed)
    return [r.standard_normal((T, N, D)) + shift for _ in range(n)]
def run(shift, pp=False):
    m = STALL(pp).fit(clips(60, 0, 0)); a = m.scores(clips(40, 0, 1)); b = m.scores(clips(40, shift, 2))
    return {k: auroc_ci(a[k], b[k], n_boot=200)[0] for k in a}
def test_shift_detected():
    for pp in (False, True):
        r = run(0.5, pp); assert r["spatial"] > 0.9 and r["joint"] > 0.9, r
def test_null_chance():
    for pp in (False, True):
        r = run(0.0, pp); assert all(abs(v - 0.5) < 0.15 for v in r.values()), r
def test_temporal_detects_dynamics():
    # same marginal N(0,I) per frame, but frames are near-copies (smooth) -> temporal diff distribution differs
    r = np.random.default_rng(3); base = r.standard_normal((40, 1, N, D))
    smooth = [b * np.ones((T, 1, 1)) + 0.1 * r.standard_normal((T, N, D)) for b in base]
    m = STALL().fit(clips(60, 0, 0)); a = m.scores(clips(40, 0, 1)); b = m.scores(smooth)
    assert auroc_ci(a["temporal"], b["temporal"], n_boot=100)[0] < 0.1 or auroc_ci(a["temporal"], b["temporal"], n_boot=100)[0] > 0.9
if __name__ == "__main__":
    print("shift0.5", run(0.5), "shift0.5 pp", run(0.5, True)); print("null", run(0.0), "null pp", run(0.0, True))
def test_dedupe_removes_duplicate_artifact():
    # real-like clips with one duplicated frame; dedupe should remove the zero-delta and make them match a clean fit
    r = np.random.default_rng(5); clean = clips(60, 0, 0)
    dup = []
    for c in clips(40, 0, 1):
        c = c.copy(); c[10] = c[9]; dup.append(c)
    keep = [np.r_[np.ones(10, bool), [False], np.ones(5, bool)] for _ in dup]
    m = STALL().fit(clean)
    plain = m.scores(dup)["temporal"]; dd = m.scores(dup, keep)["temporal"]; ref = m.scores(clips(40, 0, 2))["temporal"]
    assert abs(auroc_ci(ref, dd, n_boot=50)[0] - 0.5) < 0.2
