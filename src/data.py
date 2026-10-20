"""Clip loaders. nuScenes CAM_FRONT (samples+sweeps ~12Hz) and generic mp4."""
import json, os, numpy as np, av
from PIL import Image
def nuscenes_cam_front_clips(root, version="v1.0-mini", T=16, stride=2):
    J=lambda n: json.load(open(f"{root}/{version}/{n}.json"))
    sd=J("sample_data"); cs={c["token"]:c for c in J("calibrated_sensor")}; sen={s["token"]:s["channel"] for s in J("sensor")}
    samp={s["token"]:s["scene_token"] for s in J("sample")}; scenes={s["token"]:s["name"] for s in J("scene")}
    by={}
    for d in sd:
        if sen[cs[d["calibrated_sensor_token"]]["sensor_token"]]!="CAM_FRONT": continue
        by.setdefault(scenes[samp[d["sample_token"]]],[]).append((d["timestamp"],d["filename"]))
    for name in sorted(by):
        fr=sorted(by[name])[: T*stride : stride]
        yield name, [f"{root}/{f}" for _,f in fr], [t for t,_ in fr]
def load_images(paths): return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
def mp4_clip(path, T=16, stride=4, start=0):
    c=av.open(path); fr=[f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return np.stack(fr[start:start+T*stride:stride])
