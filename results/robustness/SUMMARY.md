# LHF-5 robustness pass (Block 6, CPU), frozen PRIMARY protocol

PRIMARY frozen: dinov3_s16, fit_resampled n=500, no dedupe, drop_pos10, joint z-sum, 1000x bootstrap. n_fit=500, n_real=150 gt. **No metric changes.**

## 1. All worldbench generators (gen0, n=150 each; Dist4D gen0 = 150 clips matching all 150 gt scenes)

| generator | n | spatial | temporal | **joint** | verdict | shortcut (excl 9->10) | flag |
|---|---|---|---|---|---|---|---|
| magicdrive | 150 | 0.630 [0.566, 0.695] | 0.844 [0.796, 0.888] | **0.804 [0.747, 0.856]** | alive | 0.653 [0.617, 0.690] | no |
| xscene | 150 | 0.477 [0.410, 0.545] | 0.813 [0.759, 0.860] | **0.725 [0.665, 0.784]** | alive | 0.653 [0.617, 0.690] | no |
| dreamforge | 150 | 0.618 [0.554, 0.683] | 0.747 [0.688, 0.801] | **0.723 [0.666, 0.784]** | alive | 0.648 [0.610, 0.686] | no |
| drivedreamer2 | 150 | 0.653 [0.589, 0.717] | 0.552 [0.485, 0.618] | **0.618 [0.556, 0.682]** | weak | 0.535 [0.484, 0.589] | no |
| dist4d | 150 | 0.544 [0.479, 0.608] | 0.550 [0.482, 0.613] | **0.568 [0.502, 0.632]** | near-chance | 0.546 [0.495, 0.597] | no |
| opendwm | 150 | 0.570 [0.504, 0.633] | 0.526 [0.460, 0.590] | **0.566 [0.502, 0.633]** | near-chance | 0.544 [0.493, 0.596] | no |

## 2. Seed robustness (joint AUROC, gen0..gen3)

| generator | gen0 | gen1 | gen2 | gen3 | mean | sd |
|---|---|---|---|---|---|---|
| magicdrive | 0.804 | 0.792 | 0.812 | 0.814 | 0.805 | 0.010 |
| drivedreamer2 | 0.618 | 0.625 | 0.622 | 0.622 | 0.622 | 0.003 |

## 3. Ranking vs published FVD

Source: WorldLens (Liang et al., CVPR 2026) Table 1, G.7 Perceptual Discrepancy = I3D FVD between real and generated; arXiv 2512.10958v2 https://arxiv.org/html/2512.10958v2 ; benchmark data = worldbench/videogen (same sample pool; their FVD is on their multi-view protocol, not our CAM_FRONT re-encode).

Real gt mean clip joint = 0.767.

| generator | mean clip joint | joint AUROC | WorldLens FVD (G.7) |
|---|---|---|---|
| magicdrive | 2.692 | 0.804 | 222.00 |
| dreamforge | 1.941 | 0.723 | 189.76 |
| xscene | 1.754 | 0.725 | 179.74 |
| drivedreamer2 | 1.492 | 0.618 | 127.07 |
| opendwm | 1.252 | 0.566 | 90.42 |
| dist4d | 1.107 | 0.568 | 58.08 |

- Spearman(mean clip joint, FVD) = **1.000** (n=6; exact permutation p = 2/720 ≈ 0.003 two-sided).
- Spearman(joint AUROC, FVD) = 0.886 (p = 0.019).
- Per-paper nuScenes FVDs use heterogeneous protocols (e.g. DriveDreamer-2 55.7 / MagicDrive 217.9 / DreamForge 209.9 / X-Scene 179.7 from the X-Scene and DD2 papers); not mixed into the Spearman. No per-paper OpenDWM nuScenes FVD found.

## 4. MagicDrive per-delta temporal AUROC (14 deltas, 9->10 dropped)

| pair | AUROC [95% CI] |
|---|---|
| 0->1 | 0.778 [0.727, 0.831] |
| 1->2 | 0.802 [0.750, 0.852] |
| 2->3 | 0.837 [0.788, 0.882] |
| 3->4 | 0.863 [0.819, 0.903] |
| 4->5 | 0.855 [0.810, 0.896] |
| 5->6 | 0.856 [0.812, 0.896] |
| 6->7 | 0.838 [0.792, 0.881] |
| 7->8 | 0.838 [0.790, 0.882] |
| 8->9 | 0.862 [0.819, 0.903] |
| 10->11 | 0.842 [0.798, 0.884] |
| 11->12 | 0.694 [0.634, 0.754] |
| 12->13 | 0.806 [0.756, 0.854] |
| 13->14 | 0.822 [0.773, 0.868] |
| 14->15 | 0.760 [0.704, 0.813] |

The signal is **spread out**: 13 of 14 deltas have AUROC ≥ 0.75 (range 0.694–0.863).
The one dip, 11->12 (0.694), coincides with the gt source-frame skip: gt position 12 = source 12 after source 11 was dropped, so the real delta there spans 2 source steps, i.e. larger real motion. It is a gt-pipeline effect, not a localized generator artifact.
