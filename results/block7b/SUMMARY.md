# Block 7b: STALL-original baseline + bridging ablation (CPU, 2026-10-20)
The frozen primary (DriveLL, `src/stall.py`) is unchanged. STALL is pinned at OmerBenHayun/STALL@bfcc603ae83b4e609681277b9b5e80e7a9497e15 and its VATEX params (`precomputed/stall_params_vatex_dino_v3.npz`) are used as released. The model is facebook/dinov3-vitl16-pretrain-lvd1689m @ ea8dc28 (box HF login: access OK). AUROC treats real gt as positive. rho = Spearman of the per-generator joint AUROC vs WorldLens FVD.
DriveLL primary reference (results/robustness, n=150): MD .804, DF .723, XS .725, DD2 .618, OpenDWM .566, Dist4D .568. **Mean .667.** AUROC-rho .94; mean-clip-score rho 1.0.

## A. STALL-original (ViT-L/16 CLS, VATEX whitening + calibration, max-spatial/min-temporal, percentile average)
**CPU smoke only: n=10 paired clips per arm.** ViT-L on CPU ran at 1.5–2.9 s/frame on the shared, loaded box, so the full ~24.8k frames would take ~10–20 h. The full run is `scripts/gpu/stall_orig.sh` (Azure T4).
| cell | MD | DF | XS | DD2 | ODWM | D4D | mean | rho |
|---|---|---|---|---|---|---|---|---|
| exact (maxmin, all 16 frames) | .34 | .30 | .35 | .26 | .37 | .27 | **.31** | .14 |
| exact + drop_pos10 | .35 | .31 | .34 | .26 | .36 | .29 | .32 | .26 |
| mean agg | .40 | .42 | .40 | .30 | .40 | .33 | .38 | .72 |
These AUROCs are **below 0.5** on both branches (e.g. DD2 temporal .18). Under VATEX statistics, real nuScenes dashcam clips look *less* likely than the generated ones. This is a calibration-domain failure, not chance. The in-domain ViT-L variant (`nusc` whitening on fit_resampled 500) needs fit-set ViT-L features and is GPU only.

## B. Bridging ablation (ViT-S/16; ViT-L too slow). STALL scoring rule unless noted; whitening + calibration = in-domain nuScenes fit (in-sample percentiles)
| backbone/feat | fit n | delta | agg | MD | DF | XS | DD2 | ODWM | D4D | mean | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S CLS (STALL preproc, n=50/arm) | 200 | L2-norm | maxmin | .58 | .63 | .56 | .55 | .55 | .56 | .57 | .77 |
| S CLS (n=50) | 200 | L2-norm | mean | .61 | .66 | .57 | .57 | .54 | .58 | .59 | .54 |
| S patch-mean (n=50) | 200 | L2-norm | maxmin | .42 | .49 | .49 | .54 | .53 | .49 | .49 | -.37 |
| S patch-mean (primary cache, n=150) | 500 | L2-norm | maxmin | .50 | .56 | .46 | .56 | .55 | .52 | .52 | -.09 |
| S patch-mean (n=150) | 500 | L2-norm | mean | .55 | .61 | .51 | .59 | .59 | .52 | .56 | .20 |
| S patch tokens (n=150) | 500 | L2-norm | maxmin | .46 | .56 | .41 | .57 | .50 | .47 | .49 | -.20 |
| S patch tokens (n=150) | 500 | L2-norm | mean | .54 | .61 | .45 | .62 | .55 | .51 | .55 | .09 |
| **S patch-mean (n=150)** | 500 | **raw** | maxmin | .85 | .83 | .85 | .66 | .60 | .62 | .73 | .77 |
| **S patch-mean (n=150)** | 500 | **raw** | mean | .89 | .81 | .86 | .63 | .61 | .62 | **.74** | .89 |
| S patch-mean +d10 (n=150) | 500 | raw | mean | .88 | .81 | .86 | .62 | .58 | .60 | .72 | .89 |
| DriveLL primary: S patch tokens, LW Mahalanobis, z-sum, d10 | 500 | raw | mean | .80 | .72 | .73 | .62 | .57 | .57 | .67 | .94 |
The patch-token raw-delta cell under STALL scoring was stopped for CPU time and not run. Numbers: `vitl.json`, `vits.json`, `patch_cache.json`, `patch_cache_rawdelta_pmean_log.json`.

### Reading (for the novelty gate in the README)
- The dominant factor is **raw (un-normalised) frame deltas**. STALL's L2 normalisation of deltas removes the motion-magnitude signal, and its temporal branch drops to chance. The second factor is **in-domain calibration**: VATEX-calibrated STALL-original scores below 0.5 here, though only at n=10 and only for ViT-L.
- **Patch tokens are not the source of the gain.** A mean-pooled patch vector with raw deltas (.74) beats full patch tokens in the DriveLL primary (.67). Mean aggregation helps the ranking (rho .89 vs .77).
- Gate condition (1): the AUROC margin over STALL-original is clearly ≥0.05 (smoke n=10: .31 vs .67). The gain is attributable to in-domain fit plus raw deltas, **not** patch tokens. The parent and Idea Agent should decide whether "in-domain fit" alone satisfies the attribution clause. Condition (2) (k≤10 bootstrap rho vs FVD) is Block 7 item 2 (other worker).

## Timings (box CPU, 8 cores, shared with Block 7 jobs; load 15–30)
ViT-S/16: 0.42 s/frame (550 clips, ~62 min). ViT-L/16: 1.5 s/frame unloaded and ~2.9 s/frame under load (74 clips).

## GPU command (Azure T4, after `hf auth login` with DINOv3 access)
`cd ~/lhf5-video-realism && git pull && bash scripts/gpu/stall_orig.sh`
The fit clips come from the existing `scripts/azure/lhf5_fit_500.tar` on the VM. The worldbench gen0/gt clips are public and are re-downloaded at revision 63217f2 and re-encoded with `scripts/reencode.sh`, so no transfer from the box is needed. Output: `results/block7b/vitl_full.json`.
