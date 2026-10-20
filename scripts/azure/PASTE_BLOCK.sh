#!/usr/bin/env bash
# ONE paste block for the Azure VM: recreates scripts/azure (no git auth needed) and runs it.
conda activate azureml_py38 2>/dev/null || source activate azureml_py38 2>/dev/null || true
mkdir -p ~/lhf5_azure && cd ~/lhf5_azure
cat > reencode.sh <<'LHF5_EOF_reencode_sh'
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
LHF5_EOF_reencode_sh
cat > check_nusc.py <<'LHF5_EOF_check_nusc_py'
#!/usr/bin/env python
"""Check nuScenes v1.0-trainval on disk: split counts + CAM_FRONT presence. Prints JSON, writes check_nusc.json."""
import os, sys, json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def cam_front_chain(nusc, scene):
    s = nusc.get('sample', scene['first_sample_token'])
    sd = nusc.get('sample_data', s['data']['CAM_FRONT'])
    while sd['prev']: sd = nusc.get('sample_data', sd['prev'])
    out = []
    while True:
        out.append(sd)
        if not sd['next']: break
        sd = nusc.get('sample_data', sd['next'])
    return out

def main():
    root = os.environ.get('NUSC', sys.argv[1] if len(sys.argv) > 1 else '.')
    nusc = NuScenes('v1.0-trainval', dataroot=root, verbose=False)
    splits = create_splits_scenes()
    train, val = set(splits['train']), set(splits['val'])
    per = {}
    exp = have = 0
    for sc in nusc.scene:
        ch = cam_front_chain(nusc, sc)
        n_ok = sum(os.path.exists(os.path.join(root, sd['filename'])) for sd in ch)
        exp += len(ch); have += n_ok
        per[sc['name']] = dict(expected=len(ch), present=n_ok,
                               keyframes=sum(sd['is_key_frame'] for sd in ch))
    full = lambda S: sorted(n for n in S if n in per and per[n]['present'] == per[n]['expected'])
    summ = dict(dataroot=root, n_scenes_meta=len(nusc.scene),
                n_train_official=len(train), n_val_official=len(val),
                n_train_in_meta=sum(n in per for n in train), n_val_in_meta=sum(n in per for n in val),
                cam_front_jpegs_expected=exp, cam_front_jpegs_present=have,
                train_fully_present=len(full(train)), val_fully_present=len(full(val)),
                train_fully_present_scenes=full(train))
    json.dump(dict(summary=summ, per_scene=per), open('check_nusc.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'train_fully_present_scenes'}, indent=1))

if __name__ == '__main__': main()
LHF5_EOF_check_nusc_py
cat > export_fit.py <<'LHF5_EOF_export_fit_py'
#!/usr/bin/env python
"""Export 500 TRAIN-scene fit clips (16 CAM_FRONT frames @~12Hz from first keyframe), same reencode pipeline.
Never uses val scenes. Output: fit_500/*.mp4 + manifest.json -> lhf5_fit_500.tar"""
import os, sys, json, random, shutil, subprocess, tempfile, tarfile
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from check_nusc import cam_front_chain

HERE = os.path.dirname(os.path.abspath(__file__))
N, T, SEED = 500, 16, 0

def ffmpeg_bin():
    if shutil.which('ffmpeg'): return 'ffmpeg'
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'imageio-ffmpeg'])
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()

def main():
    root = os.environ['NUSC']
    out = os.path.abspath(os.environ.get('OUT', 'fit_500')); os.makedirs(out, exist_ok=True)
    ff = ffmpeg_bin()
    # make reencode.sh find our ffmpeg
    bindir = tempfile.mkdtemp(); os.symlink(ff, os.path.join(bindir, 'ffmpeg'))
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ['PATH'])
    nusc = NuScenes('v1.0-trainval', dataroot=root, verbose=False)
    sp = create_splits_scenes(); train, val = set(sp['train']), set(sp['val'])
    cands = []
    for sc in nusc.scene:
        if sc['name'] not in train or sc['name'] in val: continue
        ch = cam_front_chain(nusc, sc)
        if ch and all(os.path.exists(os.path.join(root, sd['filename'])) for sd in ch):
            k0 = next(i for i, sd in enumerate(ch) if sd['is_key_frame'])
            if len(ch) - k0 >= T: cands.append((sc, ch[k0:k0 + T]))
    cands.sort(key=lambda c: c[0]['name'])
    random.Random(SEED).shuffle(cands)
    pick = sorted(cands[:N], key=lambda c: c[0]['name'])
    print(f'qualifying train scenes: {len(cands)}; exporting {len(pick)}', flush=True)
    man = []
    for i, (sc, w) in enumerate(pick):
        assert sc['name'] not in val
        tmp = tempfile.mkdtemp()
        for j, sd in enumerate(w):
            os.symlink(os.path.join(root, sd['filename']), os.path.join(tmp, f'{j:06d}.jpg'))
        mid = os.path.join(tmp, 'crf18.mp4'); dst = os.path.join(out, f"{sc['name']}.mp4")
        subprocess.check_call([ff, '-nostdin', '-loglevel', 'error', '-y', '-framerate', '12',
            '-i', os.path.join(tmp, '%06d.jpg'), '-frames:v', str(T), '-c:v', 'libx264',
            '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', mid])
        subprocess.check_call(['bash', os.path.join(HERE, 'reencode.sh'), mid, dst], env=env)
        shutil.rmtree(tmp)
        ts = [sd['timestamp'] for sd in w]
        man.append(dict(scene=sc['name'], scene_token=sc['token'], file=os.path.basename(dst),
            sample_data_tokens=[sd['token'] for sd in w], sample_tokens=[sd['sample_token'] for sd in w],
            is_key_frame=[sd['is_key_frame'] for sd in w], timestamps_us=ts,
            mean_hz=(T - 1) / ((ts[-1] - ts[0]) / 1e6)))
        if i % 50 == 0: print(f'{i}/{len(pick)}', flush=True)
    json.dump(dict(n=len(man), n_qualifying=len(cands), seed=SEED, frames=T, split='train',
                   pipeline='jpeg->h264 crf18 @12fps -> reencode.sh (12fps,16:9 crop,456x256,crf23)',
                   clips=man), open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    with tarfile.open('lhf5_fit_500.tar', 'w') as t: t.add(out, arcname='fit_500')
    print(f"wrote lhf5_fit_500.tar ({os.path.getsize('lhf5_fit_500.tar')/1e6:.1f} MB), {len(man)} clips")

if __name__ == '__main__': main()
LHF5_EOF_export_fit_py
cat > run_all.sh <<'LHF5_EOF_run_all_sh'
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
LHF5_EOF_run_all_sh
chmod +x *.sh && nohup bash run_all.sh > run_all.log 2>&1 &
tail -f run_all.log
