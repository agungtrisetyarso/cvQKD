import numpy as np, json
from scipy.stats import norm
from keyrate import keyrate, c_max
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
VA,eta,vel=4.0,0.6,0.05; eps=1e-9; z=norm.isf(eps)
def r_of(T,xi):
    VB=eta*T*(VA+xi)+1+2*vel; return eta*c_max(T,xi)/(VB+1)
def Nstar(T,xi): r=r_of(T,xi); return (z*(1+np.sqrt(1+r*r))/r)**2
def Nbound(T,xi):  # explicit upper bound used in the theorem
    return 4*z**2*((eta*T*(VA+xi)+2+2*vel)**2/(2*eta**2*T*xi*(1-T))+1)
plob=lambda T: -np.log2(1-T)
O={}; Ls=np.linspace(2,300,300)
for xi in (0.01,0.02,0.05):
    Ts=10**(-0.02*Ls)
    K=np.array([keyrate(VA,T,xi)[0] for T in Ts]); N=np.array([Nstar(T,xi) for T in Ts])
    NB=np.array([Nbound(T,xi) for T in Ts]); P=plob(Ts)
    lim=8*z**2*(1+vel)**2/(eta**2*xi*np.log(2))
    O[xi]=dict(L=Ls.tolist(),KN=(K*N).tolist(),PN=(P*N).tolist(),PNB=(P*NB).tolist(),limit=lim)
    assert np.all(N<=NB*(1+1e-12)), "bound fails"
    i80=np.argmin(abs(Ls-80)); i15=np.argmin(abs(Ls-15)); i300=-1
    print("xi",xi,"limit",lim," PLOB*N* at 15/80/300 km:",P[i15]*N[i15],P[i80]*N[i80],P[i300]*N[i300],
          " GG02 K*N* 15/80/300:",K[i15]*N[i15],K[i80]*N[i80],K[i300]*N[i300], " max PLOB*Nbound over L>=50:", (P*NB)[Ls>=50].max())
json.dump(O,open("exposure2.json","w"),indent=1,default=float)
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.linewidth":0.6})
fig,ax=plt.subplots(figsize=(3.45,2.35))
for xi,c in zip((0.01,0.02,0.05),("C0","C1","C2")):
    d=O[xi]; KN=np.array(d["KN"]); L=np.array(d["L"]); m=KN>0
    ax.semilogy(np.array(d["L"]),d["PNB"],"-",c=c,lw=1.0,label=f"bound (\\ref{{eq:expo}}), $\\xi={xi}$")
    ax.semilogy(L[m],KN[m],"--",c=c,lw=0.9)
    ax.axhline(d["limit"],c=c,ls=":",lw=0.7)
ax.set_xlabel("Fiber length $L$ (km)"); ax.set_ylabel("Secret bits per detection window")
ax.set_ylim(1e2,1e7)
from matplotlib.lines import Line2D
h=[Line2D([],[],c="C0",label="$\\xi=0.01$"),Line2D([],[],c="C1",label="$\\xi=0.02$"),Line2D([],[],c="C2",label="$\\xi=0.05$"),
   Line2D([],[],c="k",ls="-",label="capacity bound"),Line2D([],[],c="k",ls="--",label="GG02 rate"),Line2D([],[],c="k",ls=":",label="$L\\to\\infty$ limit")]
ax.legend(handles=h,fontsize=6,ncol=2,loc="upper right",framealpha=0.95)
fig.tight_layout(pad=0.3); fig.savefig("fig_exposure.pdf")
