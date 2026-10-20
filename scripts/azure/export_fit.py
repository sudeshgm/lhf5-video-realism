#!/usr/bin/env python
"""Export 500 TRAIN-scene fit clips from the first keyframe, two variants of the SAME windows:
  fit_resampled/ : PRIMARY. Nearest-timestamp sampling of raw CAM_FRONT (samples+sweeps) on a uniform 1/12 s grid (resample.py)
  fit_raw/       : secondary. 16 raw consecutive CAM_FRONT frames
Both: jpeg -> h264 crf18 @12fps -> reencode.sh. Never uses val scenes.
Output: fit_500/{fit_resampled,fit_raw}/*.mp4 + fit_500/manifest.json -> lhf5_fit_500.tar"""
import os, sys, json, random, shutil, subprocess, tempfile, tarfile
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from check_nusc import cam_front_chain
from resample import resample_indices, raw_indices

HERE = os.path.dirname(os.path.abspath(__file__))
N, T, SEED = int(os.environ.get('N_FIT', 500)), 16, 0

def ffmpeg_bin():
    if shutil.which('ffmpeg'): return 'ffmpeg'
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'imageio-ffmpeg'])
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()

VARIANTS = ('fit_resampled', 'fit_raw')

def encode(ff, env, tmp, mid, dst):
    subprocess.check_call([ff, '-nostdin', '-loglevel', 'error', '-y', '-framerate', '12',
        '-i', os.path.join(tmp, '%06d.jpg'), '-frames:v', str(T), '-c:v', 'libx264',
        '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', mid])
    subprocess.check_call(['bash', os.path.join(HERE, 'reencode.sh'), mid, dst], env=env)

def main():
    root = os.environ['NUSC']
    out = os.path.abspath(os.environ.get('OUT', 'fit_500')); os.makedirs(out, exist_ok=True)
    ff = ffmpeg_bin()
    # make reencode.sh find our ffmpeg
    bindir = tempfile.mkdtemp(); os.symlink(ff, os.path.join(bindir, 'ffmpeg'))
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ['PATH'])
    nusc = NuScenes(os.environ.get('NUSC_VERSION', 'v1.0-trainval'), dataroot=root, verbose=False)
    sp = create_splits_scenes(); train, val = set(sp['train']), set(sp['val'])
    cands = []
    for sc in nusc.scene:
        if sc['name'] not in train or sc['name'] in val: continue
        ch = cam_front_chain(nusc, sc)
        if ch and all(os.path.exists(os.path.join(root, sd['filename'])) for sd in ch):
            k0 = next(i for i, sd in enumerate(ch) if sd['is_key_frame'])
            ch = ch[k0:]
            try: idx = {'fit_resampled': resample_indices([sd['timestamp'] for sd in ch], T, 12.0),
                        'fit_raw': raw_indices(len(ch), T)}
            except ValueError: continue
            cands.append((sc, ch, idx))
    cands.sort(key=lambda c: c[0]['name'])
    random.Random(SEED).shuffle(cands)
    pick = sorted(cands[:N], key=lambda c: c[0]['name'])
    print(f'qualifying train scenes: {len(cands)}; exporting {len(pick)}', flush=True)
    man = []
    for v in VARIANTS: os.makedirs(os.path.join(out, v), exist_ok=True)
    for i, (sc, ch, idx) in enumerate(pick):
        assert sc['name'] not in val
        ent = dict(scene=sc['name'], scene_token=sc['token'], file=f"{sc['name']}.mp4", variants={})
        for v in VARIANTS:
            w = [ch[k] for k in idx[v]]
            tmp = tempfile.mkdtemp()
            for j, sd in enumerate(w):
                os.symlink(os.path.join(root, sd['filename']), os.path.join(tmp, f'{j:06d}.jpg'))
            mid = os.path.join(tmp, 'crf18.mp4'); dst = os.path.join(out, v, f"{sc['name']}.mp4")
            encode(ff, env, tmp, mid, dst)
            shutil.rmtree(tmp)
            ts = [sd['timestamp'] for sd in w]
            ent['variants'][v] = dict(path=f"{v}/{sc['name']}.mp4", chain_indices=idx[v],
                sample_data_tokens=[sd['token'] for sd in w], sample_tokens=[sd['sample_token'] for sd in w],
                is_key_frame=[sd['is_key_frame'] for sd in w], timestamps_us=ts,
                n_repeated_source=T - len(set(idx[v])),
                mean_hz=(T - 1) / ((ts[-1] - ts[0]) / 1e6))
        man.append(ent)
        if i % 50 == 0: print(f'{i}/{len(pick)}', flush=True)
    json.dump(dict(n=len(man), n_qualifying=len(cands), seed=SEED, frames=T, split='train', variants=list(VARIANTS), primary='fit_resampled',
                   resampler='nearest-timestamp on uniform 1/12 s grid from first keyframe (resample.py)',
                   pipeline='jpeg->h264 crf18 @12fps -> reencode.sh (12fps,16:9 crop,456x256,crf23)',
                   clips=man), open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    with tarfile.open('lhf5_fit_500.tar', 'w') as t: t.add(out, arcname='fit_500')
    print(f"wrote lhf5_fit_500.tar ({os.path.getsize('lhf5_fit_500.tar')/1e6:.1f} MB), {len(man)} clips")

if __name__ == '__main__': main()
