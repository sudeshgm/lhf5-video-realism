#!/usr/bin/env bash
# REAL RUN (pre-registered). Primary = first command. Usage: scripts/real_run.sh <backbone>
set -euo pipefail
BB=${1:-dinov3_s16}; E=/workspace/data/enc; F=data/fit_500
run(){ n=$1; shift; python src/stall.py --backbone $BB --real $E/gt --gen $E/magicdrive $E/drivedreamer2 --out results/real_run/${BB}_$n.json --label "REAL RUN $BB $n" "$@" > /dev/null; echo done $n; }
run primary          --fit $F/fit_resampled
run unamended        --fit $F/fit_resampled --no-drop-pos10
run amended_dedupe   --fit $F/fit_resampled --dedupe
run fitraw_primary   --fit $F/fit_raw
