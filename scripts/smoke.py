import sys,os,glob,time,json,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","src"))
from features import Backbone; from data import *
mid=sys.argv[1]; out="results/smoke"; os.makedirs(out,exist_ok=True)
t0=time.time(); bb=Backbone(mid); log={"model":mid,"load_s":round(time.time()-t0,2),"clips":[]}
jobs=[]
for i,(n,paths,ts) in enumerate(nuscenes_cam_front_clips("/workspace/data")):
    if i==5: break
    jobs.append(("real",n,lambda p=paths: load_images(p),f"~{(len(ts)-1)/((ts[-1]-ts[0])/1e6):.1f}Hz"))
for f in sorted(glob.glob("/workspace/data/gen/generation/*.mp4"))[:5]:
    jobs.append(("gen",os.path.basename(f)[:-4],lambda f=f: mp4_clip(f),"6Hz (24fps stride4)"))
T=time.time()
for lab,n,ld,hz in jobs:
    a=time.time(); fr=ld(); b=time.time(); p,c=bb(fr); e=time.time()
    np.save(f"{out}/{lab}_{n}_patch.npy",p.astype(np.float16)); np.save(f"{out}/{lab}_{n}_cls.npy",c.astype(np.float16))
    r=dict(label=lab,clip=n,src_hw=list(fr.shape[1:3]),rate=hz,patch=list(p.shape),cls=list(c.shape),load_s=round(b-a,2),fwd_s=round(e-b,2)); print(r); log["clips"].append(r)
log["total_s"]=round(time.time()-T,2); print("TOTAL",log["total_s"],"s")
json.dump(log,open(f"{out}/smoke_log.json","w"),indent=1)
