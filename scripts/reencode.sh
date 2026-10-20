#!/usr/bin/env bash
# Common domain-control pipeline for ALL arms (real fit, real held-out, gen).
# usage: reencode.sh <input.mp4 | "dir/with/frames/%06d.jpg"> <out.mp4> [in_fps]
# -> 16 frames, 12 fps, 16:9 crop, 455x256 (even width 456), H.264 High, CRF 23, preset medium, yuv420p, no B-frame tuning.
set -euo pipefail
IN="$1"; OUT="$2"; FPS="${3:-12}"
PRE=(); [[ "$IN" == *%* ]] && PRE=(-framerate "$FPS")
ffmpeg -nostdin -loglevel error -y "${PRE[@]}" -i "$IN" \
  -vf "fps=12,crop=iw:'min(ih,iw*9/16)',scale=456:256:flags=bicubic,setsar=1" \
  -frames:v 16 -c:v libx264 -profile:v high -preset medium -crf 23 -pix_fmt yuv420p -an "$OUT"
