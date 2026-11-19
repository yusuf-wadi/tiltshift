import sys, av, numpy as np, cv2
path,start,dur=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
c=av.open(path); v=c.streams.video[0]; c.seek(int(start*1e6)); prev=None; R=[]
for f in c.decode(v):
    t=float(f.pts*v.time_base)
    if t<start: continue
    if t>start+dur: break
    a=f.to_ndarray(format="rgb24",width=160,height=90).astype(np.float32)
    g=cv2.resize(a,(48,27),interpolation=cv2.INTER_AREA).ravel()
    g=(g-g.mean())/(g.std()+1e-3)
    if prev is not None: R.append((t,1-float((g*prev).mean())))
    prev=g
r=np.array(R); d=r[:,1]
print("1-ncc pct 50/90/99/99.5", np.percentile(d,[50,90,99,99.5]).round(3))
for thr in (0.3,0.4,0.5,0.6):
    print(thr,(d>thr).sum(),[round(x,1) for x in r[d>thr,0]])
