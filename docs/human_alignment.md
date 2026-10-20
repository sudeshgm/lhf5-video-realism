# Per-clip human alignment — availability check (2026-10-20)

## Existing human ratings
**WorldLens-26K** (WorldLens, arXiv:2512.10958, CVPR 2026)
- **Content:** 26,808 scoring records of generated driving videos. Each is a discrete 1–10 score plus a free-text rationale on 4 dimensions: World Realism, Physical Plausibility, 3D/4D Consistency, Behavioral Safety. Rated by 10 annotators in 2 independent groups, who viewed the video plus a semantic mask, depth and 3D boxes; divergent ratings were re-adjudicated (~128 s per annotation, >930 h). The paper says the generators are the WorldLens nuScenes set, which overlaps with our worldbench/videogen clips.
- **Availability: NOT RELEASED.** The GitHub README (https://github.com/worldbench/WorldLens) lists "Release the WorldLens-26K dataset" as an open TODO, and the section reads "To be updated". In issue #8 (https://github.com/worldbench/WorldLens/issues/8, 2026-02-28) the authors said it was under internal review, "open source in May or before". As of 2026-10-20 the Hugging Face org `worldbench` (https://huggingface.co/datasets/worldbench) has only videogen, occgen, lidargen, MIST-Bench and MIST-Train, and no 26K dataset.
- **License:** the repo code is Apache-2.0 and worldbench/videogen is tagged apache-2.0. No license has been stated for 26K. The clips derive from nuScenes, so its non-commercial terms likely apply.
- **If it is released:** compute Spearman/Kendall between our per-clip score and the World Realism score (and Physical Plausibility for the temporal term), per generator and pooled, with bootstrap CIs. Also consider emailing the authors (Ao Liang / Lingdong Kong) to request early access to the realism subset.

Others checked: FVMD (arXiv:2407.16124) and JEDi (arXiv:2410.05203) ran human studies, but on general or synthetic-noise video, not nuScenes world-model clips. ACT-Bench and WorldScore have no per-clip human realism ratings for driving. **Result: no usable public per-clip human realism ratings for nuScenes world-model clips exist right now.**

## Fallback: minimal 2AFC study (design only, not run)
- **Question:** "Which clip looks more like real dashcam footage?" (realism only; no masks or depth shown).
- **Stimuli:** our re-encoded 12 fps, 16-frame (~1.3 s) 456x256 clips, looped twice, with the frame-10 duplicate handled the same way as in the metric. Use 6 generators × 20 scenes = 120 generated clips, sampled stratified by our score (quintiles), plus the matching 20 gt clips.
- **Pairs:** (a) **gen-vs-gen, same scene** (the main test of per-clip alignment): 20 scenes × C(6,2)=15 = 300 pairs. (b) **gen-vs-real, same scene**: 120 pairs (a human "detectability" rate for each generator). Left/right order and pair order are randomised. Add 10% repeated pairs for consistency and 10 catch trials (heavily corrupted clip vs real).
- **Raters:** 15 raters, each rating all 420 pairs (about 35–40 min at ~5 s per pair; split into 2 sessions). That gives ≥15 votes per pair, enough for a pair-level majority with a ±0.25 CI. Exclude raters with <80% correct on catch trials or <70% self-consistency. Raters are recruited from Prolific or lab volunteers. The nuScenes non-commercial terms allow showing clips to raters for research.
- **Analysis:**
  1. Pair-level agreement: % of pairs where the clip with the lower (more real) metric score wins the human majority. Report with a bootstrap CI over scenes and compare against chance (50%) and inter-rater agreement (split-half ceiling).
  2. Per-clip Bradley–Terry (or TrueSkill) realism scores. Then compute Spearman/Kendall vs our score, within-scene and pooled, with a scene-cluster bootstrap.
  3. Generator level: human gen-vs-real fool rate and BT means vs our mean score and WorldLens FVD (Spearman, n=6, reported descriptively).
  4. Baselines on the same pairs: per-clip FID-style or I3D distance to the gt clip, our spatial-only and temporal-only scores, and the STALL original recipe.
- **Pre-registration:** primary = within-scene gen-vs-gen pair agreement of our joint score (DINOv3, amended protocol). Success if it is above 60% and the bootstrap CI excludes 50%.
