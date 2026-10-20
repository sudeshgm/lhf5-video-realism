#!/usr/bin/env bash
# Block 7b GPU job: STALL-original (DINOv3 ViT-L/16 CLS, VATEX whitening) on ALL 150 gt + 6x150 gen0 clips, plus the
# in-domain-whitened variant on fit_resampled 500. ~25k frames of ViT-L: ~10 h on the 8-core box CPU, ~10-20 min on a T4.
# Run from the repo root on the Azure T4 (azureml_py38). Needs: an HF login with DINOv3 access (`hf auth login`),
# and scripts/azure/lhf5_fit_500.tar from the earlier Azure export (already on the VM). The worldbench clips are public
# (Apache-2.0) and are downloaded here at the pinned revision, then re-encoded with the same scripts/reencode.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."; R=$PWD; W=${WORK:-$HOME/lhf5_b7b}; mkdir -p $W/enc
pip -q install "transformers>=4.56" av scikit-learn scipy huggingface_hub
[ -d $W/STALL ] || git clone -q https://github.com/OmerBenHayun/STALL $W/STALL; git -C $W/STALL checkout -q bfcc603ae83b4e609681277b9b5e80e7a9497e15
[ -d $W/enc/fit_resampled ] || { tar xf scripts/azure/lhf5_fit_500.tar -C $W; ln -sfn $W/fit_500/fit_resampled $W/enc/fit_resampled; }
REV=63217f288a68ba2097d84c3b160dbd43f45dcacb
declare -A P=([gt]=gt/video_submisson [magicdrive]=magicdrive/video_submisson [dreamforge]=dreamforge/video_submisson
  [xscene]=xscene/video_submission [drivedreamer2]=drivedreamer2/video_submission [opendwm]=opendwm/video_submission [dist4d]=dist4d/video_submission)
for a in "${!P[@]}"; do
  hf download worldbench/videogen --repo-type dataset --revision $REV --local-dir $W/wb --include "nuscenes/${P[$a]}/*_gen0/*_CAM_FRONT.mp4" > /dev/null
  mkdir -p $W/enc/$a; for f in $W/wb/nuscenes/${P[$a]}/*_gen0/*_CAM_FRONT.mp4; do
    o=$W/enc/$a/$(basename $(dirname $f) _gen0).mp4; [ -f $o ] || bash scripts/reencode.sh $f $o; done; echo "$a: $(ls $W/enc/$a | wc -l) clips"
done
export STALL_DIR=$W/STALL STALL_CACHE=$W/feats
python -m src.stall_orig.extract --cache $W/feats --model facebook/dinov3-vitl16-pretrain-lvd1689m --tag vitl16 $W/enc/{fit_resampled,gt,magicdrive,dreamforge,xscene,drivedreamer2,opendwm,dist4d}
python scripts/b7b_score.py vitl_full | tee $W/vitl_full.txt
echo "DONE. Paste $W/vitl_full.txt back (or: git add results/block7b/vitl_full.json && git commit -m 'b7b vitl full' && git push)."
