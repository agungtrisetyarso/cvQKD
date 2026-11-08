import numpy as np
from keyrate import *
def cond_hom(gam, keep, meas, q=0):
    idx_k=np.concatenate([[2*k,2*k+1] for k in keep]); j=2*meas+q
    A=gam[np.ix_(idx_k,idx_k)]; b=gam[j,j]; Cc=gam[idx_k,j][:,None]
    return A-(Cc@Cc.T)/b
def kh(VA,T,xi,c=0,eta=0.6,vel=0.05,beta=0.95):
    gam,gAB=state(VA,T,xi,c,eta,vel,het=False); Ks=[]
    for q in (0,1):
        gBA=cond_het(gam,[1],0); I=0.5*np.log2(gam[2+q,2+q]/gBA[q,q])
        chi=S(gAB)-S(cond_hom(gam,[0,2,3],1,q)); Ks.append(beta*I-chi)
    return np.mean(Ks)
for xi in (0.02,0.05): print("hom",xi,kh(4,0.5,xi))
# untrusted-detector homodyne (all noise to Eve): T'=eta T, xi'=xi+vel/(eta T)
for xi in (0.02,0.05): print("hom untrusted",xi,kh(4,0.3,xi+0.05/0.3,eta=1-1e-9,vel=0))
