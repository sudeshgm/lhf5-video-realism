"""PLUMBING ONLY: build fit clips from nuScenes v1.0-mini scenes NOT in the worldbench gt set.
Non-overlapping 16-frame windows of consecutive CAM_FRONT frames (~12Hz), double-encoded:
JPEG -> 1600x900 H.264 CRF18 (mimic worldbench encode) -> scripts/reencode.sh."""
import json,os,subprocess,tempfile,shutil,sys
root="/workspace/data"; out=f"{root}/enc/mini_fit"; os.makedirs(out,exist_ok=True)
toks=set(json.load(open(f"{root}/wb/tokens.json")))
J=lambda n: json.load(open(f"{root}/v1.0-mini/{n}.json"))
S={x['token']:x for x in J('sample')}; SC={x['token']:x['name'] for x in J('scene')}
excl={S[t]['scene_token'] for t in toks if t in S}
cf=[x for x in J('sample_data') if x['filename'].split('/')[1]=='CAM_FRONT']; cf.sort(key=lambda x:x['timestamp'])
n=0
for sc in sorted(SC,key=SC.get):
    if sc in excl: continue
    fr=[x['filename'] for x in cf if S[x['sample_token']]['scene_token']==sc]
    for w in range(len(fr)//16):
        d=tempfile.mkdtemp()
        for i,f in enumerate(fr[w*16:(w+1)*16]): os.symlink(f"{root}/{f}",f"{d}/{i:06d}.jpg")
        mid=f"{d}/stage1.mp4"
        subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-framerate","12","-i",f"{d}/%06d.jpg","-c:v","libx264","-crf","18","-pix_fmt","yuv420p",mid],check=True)
        subprocess.run(["scripts/reencode.sh",mid,f"{out}/{SC[sc]}_w{w:02d}.mp4"],check=True); shutil.rmtree(d); n+=1
print("fit clips",n,"excluded scenes",sorted(SC[s] for s in excl))
