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
