# Block 7 (items 1-2; items 3-7 deferred)

## Item 2: sample efficiency (gate clause 2)

k gen clips per generator (k=150: bootstrap w/ replacement), 200 resamples; real reference = all 150 gt for FVD/FD; rho vs the same metric's full-n ranking over 6 generators.

| k | ours median ρ [IQR] | FVD-I3D | FD-DINOv3 |
|---|---|---|---|
| 5 | 0.543 [0.257, 0.771] | 0.543 [0.314, 0.771] | 0.143 [-0.143, 0.429] |
| 10 | 0.657 [0.486, 0.829] | 0.771 [0.600, 0.886] | 0.429 [0.086, 0.600] |
| 25 | 0.771 [0.643, 0.886] | 0.943 [0.871, 0.943] | 0.943 [0.829, 0.943] |
| 50 | 0.943 [0.829, 0.943] | 0.943 [0.943, 1.000] | 1.000 [0.943, 1.000] |
| 150 | 0.943 [0.943, 1.000] | 1.000 [0.943, 1.000] | 1.000 [1.000, 1.000] |

**Gate clause 2 (k≤10: ours median ρ ≥ 0.9 AND FVD < 0.7): FAIL.** Ours is 0.543 (k=5) and 0.657 (k=10); FVD-I3D is 0.543 and 0.771.

## Item 1: baselines at full n (150 gt vs 150 gen0, same re-encoded clips)

| generator | ours | FVD-I3D | FID (4 fr/clip) | FD-DINOv3 | WorldLens FVD |
|---|---|---|---|---|---|
| magicdrive | 2.692 | 660.9 | 57.29 | 3.225 | 222.00 |
| dreamforge | 1.941 | 676.3 | 56.49 | 2.775 | 189.76 |
| xscene | 1.754 | 591.6 | 55.11 | 3.105 | 179.74 |
| drivedreamer2 | 1.492 | 339.1 | 37.93 | 1.448 | 127.07 |
| opendwm | 1.252 | 413.7 | 56.88 | 2.313 | 90.42 |
| dist4d | 1.107 | 226.9 | 28.62 | 0.742 | 58.08 |

Real-vs-real floor (gt 75 vs 75): FVD_I3D=619.38, FD_DINOv3=3.45, FID_inception_perframe=79.25

| metric | ρ vs WorldLens FVD | ρ vs ours |
|---|---|---|
| FVD_I3D | 0.886 | 0.886 |
| FID_inception_perframe | 0.657 | 0.657 |
| FD_DINOv3 | 0.886 | 0.886 |
| ours_mean_clip_joint | 1.000 | 1.000 |

FVD-I3D: StyleGAN-V i3d_torchscript (400-d), 150 gt vs 150 gen clips, 16 frames @456x256 resized to 224. FID: pytorch-fid InceptionV3 pool3, frames 0,4,8,12 per clip (600 vs 600). FD-DINOv3: clip vec = [mean patch token over 15 frames (pos10 dropped); mean |delta| over 14 deltas], 768-d, n=150 (cov rank-deficient).
