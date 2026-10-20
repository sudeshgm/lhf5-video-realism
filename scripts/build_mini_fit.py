"""PLUMBING ONLY: build fit clips from nuScenes v1.0-mini scenes NOT in the worldbench gt set.
Non-overlapping 16-frame windows of CAM_FRONT (samples+sweeps) from the first keyframe, double-encoded:
JPEG -> 1600x900 H.264 CRF18 (mimic worldbench encode) -> scripts/reencode.sh.
--mode resampled (default; shared src/resample.py, nearest-timestamp on 1/12 s grid) | raw (consecutive)."""
import json,os,subprocess,tempfile,shutil,sys,argparse
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","src"))
from resample import resample_indices, raw_indices
ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["resampled","raw"],default="resampled")
ap.add_argument("--out",default=None); a=ap.parse_args()
root="/workspace/data"; out=a.out or f"{root}/enc/mini_fit_{a.mode}"; os.makedirs(out,exist_ok=True)
toks=set(json.load(open(f"{root}/wb/tokens.json")))
J=lambda n: json.load(open(f"{root}/v1.0-mini/{n}.json"))
S={x['token']:x for x in J('sample')}; SC={x['token']:x['name'] for x in J('scene')}
excl={S[t]['scene_token'] for t in toks if t in S}
cf=[x for x in J('sample_data') if x['filename'].split('/')[1]=='CAM_FRONT']; cf.sort(key=lambda x:x['timestamp'])
n=0; man=[]
for sc in sorted(SC,key=SC.get):
    if sc in excl: continue
    fr=[x for x in cf if S[x['sample_token']]['scene_token']==sc]
    k0=next(i for i,x in enumerate(fr) if x['is_key_frame']); fr=fr[k0:]
    s=0; w=0
    while True:
        ch=fr[s:]
        try: idx=resample_indices([x['timestamp'] for x in ch]) if a.mode=="resampled" else raw_indices(len(ch))
        except ValueError: break
        d=tempfile.mkdtemp()
        for i,k in enumerate(idx): os.symlink(f"{root}/{ch[k]['filename']}",f"{d}/{i:06d}.jpg")
        mid=f"{d}/stage1.mp4"; name=f"{SC[sc]}_w{w:02d}.mp4"
        subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-framerate","12","-i",f"{d}/%06d.jpg","-c:v","libx264","-crf","18","-pix_fmt","yuv420p",mid],check=True)
        subprocess.run(["scripts/reencode.sh",mid,f"{out}/{name}"],check=True); shutil.rmtree(d)
        man.append(dict(file=name,chain_start=s,indices=idx,n_repeated_source=16-len(set(idx)))); n+=1; w+=1
        s+=max(idx)+1
json.dump(man,open(f"{out}/manifest.json","w"),indent=0)
print(a.mode,"fit clips",n,"src-repeats",sum(m['n_repeated_source'] for m in man),"excluded",sorted(SC[s] for s in excl))
