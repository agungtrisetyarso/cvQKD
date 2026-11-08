import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
R=json.load(open("results2.json")); R4=json.load(open("results4.json")); marg=R4["marg"]; r0=R["link15"]["r"]; R["marg_bound_auc"]=R4["marg_bound_auc"]; R["Astar"]=R4["Astar"]
def drift(r,rt): return -0.5*np.log(1-r*r)+r*(rt-r)/(1-r*r)
plt.rcParams.update({"font.size":8,"font.family":"serif","axes.linewidth":0.6,"lines.linewidth":1.1})
fig,ax=plt.subplots(1,3,figsize=(7.16,2.4))
a=ax[0]; Nv=[m[0] for m in marg]
a.semilogx(Nv,[m[4] for m in marg],"o-",ms=3,label="Covariance $\\hat C$")
a.semilogx(Nv,[m[3] for m in marg],"s-",ms=3,label="Marg. 8 feat., QDA")
a.semilogx(Nv,[m[2] for m in marg],"D-",ms=3,label="Marg. mean+var., QDA")
a.semilogx(Nv,[m[1] for m in marg],"^-",ms=3,label="Marg. 8 feat., linear")
a.axhline(R["marg_bound_auc"],ls=":",c="k",lw=0.8); a.axhline(R["Astar"],ls="--",c="gray",lw=0.8)
a.text(4100,R["marg_bound_auc"]-0.035,"$\\chi^2$ bound",fontsize=6.5,ha="right")
a.text(4100,R["Astar"]+0.012,"$\\mathcal{A}^\\star$",fontsize=6.5,color="gray",ha="right")
a.set_xticks(Nv); a.set_xticklabels([str(n) for n in Nv],fontsize=6.5); a.xaxis.set_minor_formatter(NullFormatter())
a.set_ylim(0.42,1.24); a.set_yticks([0.5,0.6,0.7,0.8,0.9,1.0]); a.set_xlabel("Block length $N$"); a.set_ylabel("AUC")
a.legend(fontsize=5.6,loc="upper center",ncol=2,framealpha=0.95,handlelength=1.4,columnspacing=0.6)
a.set_title("(a) Bob-side monitor, $r=0.42$, $V_h=1.653$",fontsize=8)
a=ax[1]
for xi in ("0.01","0.02","0.05"):
    arr=np.array(R["dist"][xi]); ln,=a.semilogy(arr[:,0],arr[:,1],"-",label=f"$\\xi={xi}$")
    a.semilogy(arr[:,0],arr[:,2],"--",c=ln.get_color(),lw=0.8)
a.set_xlabel("Fiber length $L$ (km)"); a.set_ylabel("Pulses, $\\alpha=\\beta=10^{-9}$")
a.legend(fontsize=6,loc="upper right"); a.set_title("(b) PE-consistent budget",fontsize=8)
a=ax[2]; cu=np.array(R["cusum"]); hh=np.linspace(cu[0,0],cu[-1,0],50)
a.plot(hh,hh/drift(r0,r0)/1e5,"-",c="C0",lw=0.8); a.errorbar(cu[:,0],cu[:,1]/1e5,yerr=2*cu[:,2]/1e5,fmt="o",c="C0",ms=3,label="matched, $r'=r$")
a.plot(hh,hh/drift(r0,0.75*r0)/1e5,"-",c="C3",lw=0.8); a.errorbar(cu[:,0],cu[:,4]/1e5,yerr=2*cu[:,5]/1e5,fmt="s",c="C3",ms=3,label="mismatched, $r'=0.75r$")
a.set_xlabel("CUSUM threshold $h$"); a.set_ylabel("Delay ($10^5$ pulses)"); a.legend(fontsize=6,loc="upper left")
a.set_title("(c) Quickest detection, 15 km",fontsize=8)
fig.tight_layout(pad=0.4); fig.savefig("fig_detect.pdf")
