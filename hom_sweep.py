import itertools, json, numpy as np
from hom import kh
from keyrate import c_max
grid=list(itertools.product([0.9,0.5,0.1,0.01],[0.01,0.05,0.1],[2.0,4.0,10.0],[(0.6,0.05),(1-1e-9,0.0)]))
mind=1e9; at=None; n=0
for T,xi,VA,(eta,vel) in grid:
    K0=kh(VA,T,xi,0,eta,vel); cm=c_max(T,xi)
    for f in (-1,-0.5,0.5,0.999):
        d=kh(VA,T,xi,f*cm,eta,vel)-K0; n+=1
        if d<mind: mind,at=d,(T,xi,VA,eta,vel,f)
print("homodyne-switching sweep points",n,"min diff",mind,at)
json.dump(dict(points=n,min_diff=mind,at=at),open("hom_sweep.json","w"))
