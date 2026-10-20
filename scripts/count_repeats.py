"""Count repeated frames per clip: consecutive-frame MAE < thr (default 1/255 of range -> 1.0 on 0-255 scale)."""
import sys,glob,json,numpy as np,av,collections
def mae_seq(p):
    f=[x.to_ndarray(format='rgb24').astype(np.float32) for x in av.open(p).decode(video=0)]
    return [float(np.abs(f[i+1]-f[i]).mean()) for i in range(len(f)-1)]
if __name__=="__main__":
    thr=1.0; out={}
    for name,pat in [a.split('=',1) for a in sys.argv[1:]]:
        fs=sorted(glob.glob(pat)); rep=[]; pos=collections.Counter()
        for p in fs:
            m=mae_seq(p); r=[i+1 for i,v in enumerate(m) if v<thr]; rep.append(len(r)); pos.update(r)
        c=collections.Counter(rep); out[name]=dict(n=len(fs),clips_with_repeat=sum(x>0 for x in rep),total_repeats=sum(rep),hist=dict(sorted(c.items())),positions=dict(sorted(pos.items())))
        print(name,out[name],flush=True)
    json.dump(out,open('/workspace/lhf5/results/repeat_counts.json','w'),indent=1)
