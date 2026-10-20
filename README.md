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
