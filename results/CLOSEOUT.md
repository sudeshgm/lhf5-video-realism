# LHF-5 / DriveLL: CLOSEOUT (2026-10-20 16:01 IST)

**Status: CLOSED.** The user approved the closeout. The novelty gate was not met.

## Hypothesis
A training-free, STALL-based likelihood score on frozen DINOv3 ViT-S/16 patch features distinguishes real from world-model-generated driving video. It is fit on real nuScenes CAM_FRONT only, with a Ledoit-Wolf Gaussian, mean Mahalanobis distance, and spatial, temporal and joint (z-sum) scores. Generated videos come from worldbench/videogen; the real held-out set is 150 worldbench `gt` clips (nuScenes val).

## Pre-registered rules and amendments (box clock, IST)
| item | commit | time | content |
|---|---|---|---|
| Initial stop rule | 01cbf48 | 2026-10-20 09:35 | AUROC (real held-out vs gen) on spatial, temporal, joint: ≥0.70 alive; 0.60–0.70 weak; <0.60 → rerun once with DINOv2 ViT-B/14; if still <0.60 → DEAD |
| Deviations logged | d31e5eb | 10:02 | held-out n=150 (not 200); 12 fps, 16 frames |
| Primary metric fixed | cb593df | 10:21 | DINOv3-S/16 joint AUROC, fit_resampled, no dedupe |
| **Amendment 1: drop-pos10** | **63e16d5** | **10:23 (stamped 10:23:24)** | drop frame 10 (spatial) and delta 9→10 (temporal) in all arms, plus the n-repeats shortcut baseline; committed before any DINOv3 number |
| **Novelty gate** | **513f6ea** | **13:42 (stamped 13:42:07)** | KEEP if (1) ≥0.05 mean AUROC over STALL-original, attributable to patch tokens and/or in-domain fit; OR (2) at k≤10, median ρ ≥0.9 while FVD <0.7; else CLOSE |

## Kill-test verdict (commit 6fb06f0): NOT DEAD
Settings: DINOv3-S/16, fit_resampled n=500 (nuScenes train), 150 gt, 1000× bootstrap.
- **MagicDrive gen0: ALIVE**, joint 0.804 [0.747, 0.856].
- **DD2 gen0: WEAK**, joint 0.618 [0.556, 0.682].
- The un-amended runs were flagged: the n-repeats shortcut alone scored 1.000 for MagicDrive and 0.810 for DD2.

## Robustness (commit 1dfd62f; same protocol)
| generator | spatial | temporal | joint |
|---|---|---|---|
| MagicDrive | 0.631 | 0.844 | 0.804 |
| X-Scene | 0.477 | 0.813 | 0.725 |
| DreamForge | 0.618 | 0.747 | 0.723 |
| DriveDreamer-2 | 0.653 | 0.552 | 0.618 |
| Dist4D | 0.544 | 0.550 | 0.568 |
| OpenDWM | 0.570 | 0.527 | 0.566 |

- **Seeds** (joint, gen0–3): MagicDrive 0.806 ± 0.010; DD2 0.622 ± 0.003.
- **Ranking:** Spearman of mean clip joint vs WorldLens FVD (CVPR 2026, Table 1 G.7) = **1.000** (n=6; exact p ≈ 0.003). Joint AUROC vs FVD gives ρ = 0.886. At full n, I3D-FVD and DINOv3-FD both give 0.886 and per-frame FID gives 0.657 (commit d6148ed).
- **Per-delta:** the MagicDrive temporal signal is spread over all 14 deltas.

## Block 7b ablation (commit 382a5ae; follow-up 4d3aee1 is unvalidated)
- **Drivers:** the gain comes from the **in-domain nuScenes fit + raw (un-normalised) frame differences**, **not patch tokens**. A mean-pooled ViT-S vector with raw deltas scored mean AUROC 0.74, above the patch-token primary at 0.67. STALL's L2-normalised deltas reduce the temporal branch to chance.
- **STALL-original** (ViT-L CLS, VATEX calibration) was only a **10-clip CPU smoke test** (mean AUROC 0.31, below 0.5, meaning a calibration-domain failure). The **in-domain ViT-L variant was never run** (it is GPU-only; `scripts/gpu/stall_orig.sh` exists but was not executed).

## Sample efficiency (Block 7 item 2, commit a69a574 / d6148ed)
Median ρ [IQR] of the 6-generator ranking vs full n, 200 bootstraps:

| k | ours | I3D-FVD | DINOv3-FD |
|---|---|---|---|
| 5 | 0.543 [0.257, 0.771] | 0.543 [0.314, 0.771] | 0.143 [−0.143, 0.429] |
| 10 | 0.657 [0.486, 0.829] | 0.771 [0.600, 0.886] | 0.429 [0.086, 0.600] |
| 25 | 0.771 | 0.943 | 0.943 |
| 50 | 0.943 | 0.943 | 1.000 |
| 150 | 0.943 | 1.000 | 1.000 |

I3D-FVD ≥ ours at every k. **Gate clause 2 FAILED.**

## Novelty gate outcome: **CLOSED**
- **Clause 1:** the margin over STALL-original is untested beyond a 10-clip smoke run, and the gain is attributed to the in-domain fit plus raw differences, not to the method's patch-token design.
- **Clause 2:** failed.

## Reusable assets
- **worldbench/videogen `gt` confound:** frame 9 is duplicated at position 10 (and source frame 11 skipped) in **150/150** CAM_FRONT gt clips (rev 63217f28). Any real-vs-gen temporal metric on this benchmark is inflated by it: a repeat-count shortcut alone separates gt from MagicDrive with AUROC 1.000. See `results/repeat_counts.json` and `results/resampler_validation.json`.
- **Fit set:** `lhf5_fit_500.tar` on the **private** HF dataset `sudeshgm/lhf5-fit-500`: 500 nuScenes train scenes × {fit_resampled, fit_raw}, plus a manifest. It **must stay private** under the nuScenes non-commercial terms.
- **Pipeline:** `scripts/reencode.sh` (common H.264 domain control), `src/resample.py` (nearest-timestamp 12 Hz resampler, shared with `scripts/azure/export_fit.py` and `PASTE_BLOCK.sh`), `src/stall.py` (Ledoit-Wolf likelihood scoring with drop-pos10, dedupe and the shortcut baseline), `src/baselines.py` (I3D-FVD, FID, DINOv3-FD and a fast fixed-reference Fréchet).

## Possible salvage
A short note or workshop paper on the **frame-duplication confound in worldbench/videogen gt**: how it inflates temporal realism metrics, the trivial shortcut, and the drop-pos10 / dedupe fix. Supporting evidence: the un-amended vs amended runs and the per-delta AUROC dip at 11→12, which matches the gt skip.
