# LHF-5: Training-free likelihood scoring for real vs generated driving video

## Hypothesis
A training-free, STALL-style likelihood score computed on **frozen DINOv3 ViT-S/16** features
(fit a density model on real-video features only) distinguishes real from world-model-generated
driving video. Three scores: **spatial** (per-frame patch features), **temporal** (frame-to-frame
feature dynamics), and **joint**.

## Experiment -1 (kill test)
- Fit on 500 real CAM_FRONT clips; evaluate on 200 held-out real vs >=200 generated clips.
- 16 frames per clip @224px, ~6 Hz.

## Pre-registered stop rule
AUROC (real held-out vs gen) on spatial, temporal, joint scores:
- **>= 0.70** -> alive
- **0.60-0.70** -> weak
- **< 0.60** (near chance) -> rerun once with DINOv2 ViT-B/14; if still < 0.60 -> **DEAD**.

DINOv2 ViT-S/14 is the pre-registered alternative backbone; any switch must be explicit.

## Layout
src/ (features, data loaders) · scripts/ (backbone_check, smoke) · configs/ · results/

## Logged deviations from pre-registration
- **2026-10-20 (Block 2):** real held-out = **150** worldbench `gt` clips (one per nuScenes val scene), not 200. Reason: the worldbench/videogen gt set covers exactly 150 val scenes, and nuScenes trainval is on hold. The stop-rule thresholds are unchanged.
- **2026-10-20:** clips are 16 frames at **12 fps** (about 1.25 s) to match the generators, replacing the earlier "~6 Hz". All arms go through `scripts/reencode.sh` (456x256, H.264 CRF 23).
- Gen source: `worldbench/videogen` (MagicDrive, DriveDreamer-2; gen0, 150 clips each), with the HF revision pinned in `configs/exp_m1.yaml`.

## Primary metric (fixed 2026-10-20, Block 4)
**PRIMARY: DINOv3 ViT-S/16 joint AUROC** (worldbench gt held-out vs gen), with the fit set = `fit_resampled`: 500 train scenes, nearest-timestamp sampling of raw CAM_FRONT onto a uniform 1/12 s grid from the first keyframe (`src/resample.py`), no dedupe.
The stop rule is applied to this number only. Everything else is **secondary or sensitivity**: spatial and temporal scores, `fit_raw`, `--dedupe`, `--per-position`, other generators.

## Known confound: repeated frames (2026-10-20)
`results/repeat_counts.json` counts consecutive frames with MAE < 1/255:
- worldbench `gt` (real): **150/150 clips contain a repeat, always at frame 10** (frame 9 duplicated, source frame 11 skipped). This is a pipeline artifact; the nuScenes timestamps don't explain it.
- MagicDrive gen0: **0/150 clips with any repeat**.
- DriveDreamer-2 gen0: 24/150 raw (37/150 after re-encode); these are static-content stretches, not a fixed-position duplicate.

A real-only fixed duplicate can inflate the **temporal** score against MagicDrive. Mitigations:
- the `--dedupe` sensitivity run (drops repeated frames in ALL arms before the temporal deltas);
- the resampled fit set.
Report both alongside the primary.
