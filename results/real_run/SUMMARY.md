# LHF-5 REAL RUN: DINOv3 ViT-S/16 (2026-10-20, IST)

- **Fit:** data/fit_500 (500 nuScenes TRAIN scenes; no val/gt overlap). **Held-out:** 150 worldbench gt (val). **Gen:** MagicDrive gen0 150, DD2 gen0 150. worldbench rev 63217f28.
- **PRIMARY** = row 'primary', joint column (fit_resampled, no dedupe, drop-pos10 ON). AUROC has a 1000x bootstrap 95% CI. The shortcut column is the n-repeats AUROC, excluding pair 9->10 when drop-pos10 is ON.
- **Stop rule:** README names no single primary generator, so it is applied to both. MagicDrive 0.804 -> **ALIVE**; DD2 0.618 -> **WEAK**. Neither is <0.60, so the dinov2_b14 rerun is **not triggered**.

| run | gen | n_fit | spatial | temporal | joint | shortcut | FLAG joint<=shortcut |
|---|---|---|---|---|---|---|---|
| primary | magicdrive | 500 | 0.630 [0.566, 0.695] | 0.844 [0.796, 0.888] | **0.804 [0.747, 0.856]** (alive) | 0.653 [0.617, 0.690] | no |
| primary | drivedreamer2 | 500 | 0.653 [0.589, 0.717] | 0.552 [0.485, 0.618] | **0.618 [0.556, 0.682]** (weak) | 0.535 [0.484, 0.589] | no |
| unamended | magicdrive | 500 | 0.629 [0.564, 0.694] | 0.865 [0.820, 0.906] | **0.818 [0.763, 0.869]** (alive) | 1.000 [1.000, 1.000] | YES |
| unamended | drivedreamer2 | 500 | 0.654 [0.591, 0.718] | 0.573 [0.507, 0.637] | **0.632 [0.570, 0.694]** (weak) | 0.810 [0.753, 0.864] | YES |
| amended_dedupe | magicdrive | 500 | 0.630 [0.566, 0.695] | 0.837 [0.787, 0.882] | **0.801 [0.744, 0.854]** (alive) | 0.653 [0.617, 0.690] | no |
| amended_dedupe | drivedreamer2 | 500 | 0.653 [0.589, 0.717] | 0.553 [0.485, 0.617] | **0.621 [0.558, 0.686]** (weak) | 0.535 [0.484, 0.589] | no |
| fitraw_primary | magicdrive | 500 | 0.630 [0.566, 0.696] | 0.844 [0.796, 0.888] | **0.803 [0.747, 0.856]** (alive) | 0.653 [0.617, 0.690] | no |
| fitraw_primary | drivedreamer2 | 500 | 0.652 [0.589, 0.717] | 0.552 [0.485, 0.618] | **0.618 [0.556, 0.681]** (weak) | 0.535 [0.484, 0.589] | no |

Un-amended runs are FLAGGED: joint <= the n-repeats shortcut (1.000 MagicDrive, 0.810 DD2). This is the gt frame-9 duplicate; the amendment removes it.
