#!/usr/bin/env bash
# Block 7b GPU job (Azure T4, conda env azureml_py38, existing HF login with DINOv3 access). Self-contained and nohup-safe.
# Re-executes itself under nohup, logs to ~/stall_orig_gpu.log, writes results/block7b/vitl_full.json, then tries git push.
# Full n=150 gt vs 6 gen0 generators, ViT-L/16:
#   (a) STALL-original exact (CLS, VATEX whitening + calibration, max/min, percentile avg)
#   (b) STALL-original with IN-DOMAIN whitening/calibration on fit_resampled 500 (= gate baseline)
#   (c) patch variants with raw diffs: full patch tokens (max/min + mean), pooled patch (max/min + mean)
# Env: LIMIT=N (clips/arm, dry run), WORK (default ~/lhf5_b7b), NO_PUSH=1.
set -euo pipefail
LOG=${LOG:-$HOME/stall_orig_gpu.log}
if [ -z "${B7B_CHILD:-}" ]; then B7B_CHILD=1 nohup bash "$0" "$@" > "$LOG" 2>&1 & echo "started (pid $!); follow with: tail -f $LOG"; exit 0; fi
cd "$(dirname "$0")/../.."; R=$PWD; W=${WORK:-$HOME/lhf5_b7b}; LIMIT=${LIMIT:-0}; mkdir -p $W/enc
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null && conda activate azureml_py38 2>/dev/null || true
echo "== $(date) start; python=$(which python); LIMIT=$LIMIT"
python -c "import transformers,av,sklearn,scipy,huggingface_hub; from transformers import DINOv3ViTModel" 2>/dev/null || pip install -q -U "transformers>=4.56" av scikit-learn scipy huggingface_hub
command -v ffmpeg >/dev/null || { conda install -y -q -c conda-forge ffmpeg >/dev/null 2>&1 || pip install -q imageio-ffmpeg; command -v ffmpeg >/dev/null || ln -sf "$(python -c 'import imageio_ffmpeg as i;print(i.get_ffmpeg_exe())')" "$HOME/.local/bin/ffmpeg"; export PATH=$HOME/.local/bin:$PATH; }
python -c "import torch;print('cuda', torch.cuda.is_available())"
[ -d $W/STALL ] || git clone -q https://github.com/OmerBenHayun/STALL $W/STALL; git -C $W/STALL checkout -q bfcc603ae83b4e609681277b9b5e80e7a9497e15
TAR=${FIT_TAR:-$R/scripts/azure/lhf5_fit_500.tar}
[ -d $W/fit_500/fit_resampled ] || tar xf "$TAR" -C $W
pick(){ mkdir -p $2; ls $1/*.mp4 | sort | head -n ${LIMIT/#0/100000} | while read f; do ln -sf $f $2/; done; }
REV=63217f288a68ba2097d84c3b160dbd43f45dcacb
declare -A P=([gt]=gt/video_submisson [magicdrive]=magicdrive/video_submisson [dreamforge]=dreamforge/video_submisson
  [xscene]=xscene/video_submission [drivedreamer2]=drivedreamer2/video_submission [opendwm]=opendwm/video_submission [dist4d]=dist4d/video_submission)
for a in "${!P[@]}"; do
  hf download worldbench/videogen --repo-type dataset --revision $REV --local-dir $W/wb --include "nuscenes/${P[$a]}/*_gen0/*_CAM_FRONT.mp4" > /dev/null
  mkdir -p $W/enc/$a; n=0; for f in $(ls $W/wb/nuscenes/${P[$a]}/*_gen0/*_CAM_FRONT.mp4 | sort); do
    [ "$LIMIT" != 0 ] && [ $n -ge $LIMIT ] && break; n=$((n+1))
    o=$W/enc/$a/$(basename $(dirname $f) _gen0).mp4; [ -f $o ] || bash scripts/reencode.sh $f $o; done; echo "$a: $(ls $W/enc/$a | wc -l) clips"
done
rm -rf $W/use; pick $W/fit_500/fit_resampled $W/use/fit_resampled; for a in "${!P[@]}"; do pick $W/enc/$a $W/use/$a; done
export STALL_DIR=$W/STALL STALL_CACHE=$W/feats; G="magicdrive dreamforge xscene drivedreamer2 opendwm dist4d"
python -m src.stall_orig.extract --cache $W/feats --model facebook/dinov3-vitl16-pretrain-lvd1689m --tag vitl16 $W/use/fit_resampled $W/use/gt $(for g in $G; do echo $W/use/$g; done)
python -m src.stall_orig.patch_cells --model facebook/dinov3-vitl16-pretrain-lvd1689m --tag vitl16 --fit $W/use/fit_resampled --real $W/use/gt \
  --gen $(for g in $G; do echo $W/use/$g; done) --out results/block7b/vitl_patch_cells.json
OUTF=results/block7b/vitl_full.json; [ "$LIMIT" != 0 ] && OUTF=results/block7b/vitl_dryrun_L$LIMIT.json
B7B_OUT=$OUTF python scripts/b7b_score.py vitl_full
echo "== $(date) DONE -> $OUTF"
if [ "$LIMIT" = 0 ] && [ -z "${NO_PUSH:-}" ]; then
  git add $OUTF results/block7b/vitl_patch_cells.json && git -c user.name=lhf5-azure -c user.email=lhf5@azure.local commit -qm "Block 7b: ViT-L full STALL-original + in-domain + patch cells (Azure T4)" \
   && git pull --rebase -q && git push -q && echo "PUSHED $(git rev-parse --short HEAD)" \
   || { echo "git push failed. Paste everything between the markers back into chat:"; echo "=====JSON BEGIN====="; cat $OUTF; echo "=====JSON END====="; }
fi
