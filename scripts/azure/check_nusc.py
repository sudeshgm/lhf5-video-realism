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
