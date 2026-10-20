"""STALL-original feature extraction (Block 7b). Mirrors OmerBenHayun/STALL@bfcc603 src/stall.py:
torchvision ToTensor -> Resize((224,224)) (squash, no crop) -> ImageNet norm; feature = final-norm CLS token
(dinov3 hub forward == x_norm_clstoken == HF DINOv3ViTModel pooler_output). Also saves the mean of the
final-norm patch tokens (registers dropped) for the bridging ablation. Cache: <cache>/<tag>/<arm>__<clip>.npz
usage: python -m src.stall_orig.extract --model facebook/dinov3-vitl16-pretrain-lvd1689m --tag vitl16 DIR [DIR...]"""
import argparse, glob, os, sys, time, numpy as np, torch, av
from torchvision import transforms
from PIL import Image
from transformers import AutoModel
TF = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
def frames(p):
    fr = [f.to_ndarray(format="rgb24") for f in av.open(p).decode(video=0)]; assert len(fr) == 16, (p, len(fr)); return fr
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", default="/workspace/data/feats/stall_orig"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("dirs", nargs="+"); a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NT", os.cpu_count())))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; m = AutoModel.from_pretrained(a.model).eval().to(dev); nreg = getattr(m.config, "num_register_tokens", 0) or 0
    out = os.path.join(a.cache, a.tag); os.makedirs(out, exist_ok=True)
    paths = [p for d in a.dirs for p in sorted(glob.glob(os.path.join(d, "*.mp4")))]
    if a.limit: paths = paths[:a.limit]
    t0 = time.time(); n = 0
    for i, p in enumerate(paths):
        c = os.path.join(out, os.path.basename(os.path.dirname(p)) + "__" + os.path.basename(p) + ".npz")
        if os.path.exists(c): continue
        x = torch.stack([TF(Image.fromarray(f)) for f in frames(p)])
        with torch.no_grad(): h = m(pixel_values=x.to(dev))
        np.savez(c, cls=h.pooler_output.cpu().numpy().astype(np.float32),
                 pmean=h.last_hidden_state[:, 1 + nreg:].mean(1).cpu().numpy().astype(np.float32)); n += 1
        if n % 25 == 0: print(f"{i+1}/{len(paths)} {(time.time()-t0)/n/16:.3f}s/frame", flush=True)
    print(f"done {n} clips, {(time.time()-t0)/max(n,1)/16:.3f}s/frame", flush=True)
if __name__ == "__main__": main()
