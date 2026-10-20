"""Frozen ViT feature extraction: frames [T,H,W,3] uint8 -> patch feats [T,N,D] + CLS [T,D]."""
import numpy as np, torch
from transformers import AutoModel
from PIL import Image
MEAN=np.array([0.485,0.456,0.406],dtype=np.float32); STD=np.array([0.229,0.224,0.225],dtype=np.float32)
def prep(img, size=224):
    im=Image.fromarray(img); w,h=im.size; s=size/min(w,h)
    im=im.resize((round(w*s),round(h*s)),Image.BICUBIC); w,h=im.size
    l,t=(w-size)//2,(h-size)//2; im=im.crop((l,t,l+size,t+size))
    return ((np.asarray(im,dtype=np.float32)/255-MEAN)/STD).transpose(2,0,1)
class Backbone:
    def __init__(self, model_id):
        self.m=AutoModel.from_pretrained(model_id).eval()
        for p in self.m.parameters(): p.requires_grad=False
        self.nreg=getattr(self.m.config,"num_register_tokens",0) or 0
    @torch.no_grad()
    def __call__(self, frames):
        x=torch.from_numpy(np.stack([prep(f) for f in frames]))
        h=self.m(pixel_values=x).last_hidden_state
        return h[:,1+self.nreg:].numpy(), h[:,0].numpy()
