# PLUMBING ONLY - not a result
DINOv2-S/14 (not the pre-registered backbone). The fit set is 85 windows from only 6 v1.0-mini scenes, so the in-sample fit z-stats are tiny and unreliable.
These numbers only show that the pipeline runs end to end. Do not apply the stop rule to them.

## Block 2b (2026-10-20): DINOv3 ViT-S/16 plumbing dry run
`b2b_dinov3_s16_{primary,unamended,amended_dedupe}.json` use the pre-registered backbone, but the fit set is still the same 85-window / 6-scene v1.0-mini `fit_resampled` set. **PLUMBING ONLY - not a result. Do not apply the stop rule.**
Backbone check: last_hidden (1, 201, 384) = 1 CLS + 4 register + 196 patch tokens (14x14 at 224px); patch feats [16, 196, 384]; ~0.030 s/frame on box CPU (batch 1).
