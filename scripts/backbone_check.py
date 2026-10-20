import time,torch,sys
from transformers import AutoModel
mid=sys.argv[1] if len(sys.argv)>1 else "facebook/dinov3-vits16-pretrain-lvd1689m"
m=AutoModel.from_pretrained(mid).eval()
for p in m.parameters(): p.requires_grad=False
x=torch.randn(1,3,224,224)
with torch.no_grad():
    o=m(pixel_values=x); ts=[]
    for _ in range(10):
        t=time.time(); o=m(pixel_values=x); ts.append(time.time()-t)
h=o.last_hidden_state
print(mid,"last_hidden",tuple(h.shape),"CLS",tuple(h[:,0].shape),"sec/frame",sum(ts)/len(ts))
print("config reg tokens:",getattr(m.config,"num_register_tokens",0))
