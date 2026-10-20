#!/usr/bin/env bash
# Run on the Azure VM (conda env azureml_py38). Check -> export -> upload.
set -euo pipefail
cd "$(dirname "$0")"
: "${NUSC:=/mnt/batch/tasks/shared/LS_root/mounts/clusters/mirashi-t4-or-rbadas/code/Users/miu3kor/vjepa-kit/data/nuscenes}"
export NUSC
pip install -q nuscenes-devkit imageio-ffmpeg
python check_nusc.py
python export_fit.py
# Upload: GitHub release asset if gh is authed, else private HF dataset if HF is logged in.
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh release create fit-500 lhf5_fit_500.tar check_nusc.json -R sudeshgm/lhf5-video-realism \
     --title "fit-500 (temp transfer)" --notes "temporary transfer; delete after pull" \
  || gh release upload fit-500 lhf5_fit_500.tar check_nusc.json -R sudeshgm/lhf5-video-realism --clobber
  echo "UPLOADED: github release fit-500"
elif python -c "import huggingface_hub as h; assert h.get_token()" 2>/dev/null; then
  python - <<'PY'
from huggingface_hub import HfApi
a = HfApi(); r = a.whoami()['name'] + '/lhf5-fit-500'
a.create_repo(r, repo_type='dataset', private=True, exist_ok=True)
for f in ['lhf5_fit_500.tar', 'check_nusc.json']: a.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=r, repo_type='dataset')
print('UPLOADED: hf dataset', r)
PY
else
  echo "NO UPLOAD AUTH. Tar is at: $PWD/lhf5_fit_500.tar"
  echo "Either: 'huggingface-cli login' (paste your HF token, masked) then rerun the upload, or download the tar via the Azure ML Studio file browser."
fi
