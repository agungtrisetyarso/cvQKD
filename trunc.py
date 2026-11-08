import numpy as np, json
from scipy import integrate
V=2.25; k=1/0.674; rng=np.random.default_rng(3); M=4_000_000
x0=rng.standard_normal((M,2))*np.sqrt(V); y1=rng.gamma(k,V/k,size=M); x1=rng.standard_normal((M,2))*np.sqrt(y1)[:,None]
z0=(x0[:,0]*x0[:,1])**2; z1=(x1[:,0]*x1[:,1])**2
res=[]
for tau in (0.5,1,1.5,2,3,4,5,6,8,10):
    g=np.minimum(z1,tau).mean()-np.minimum(z0,tau).mean(); se=np.sqrt(np.minimum(z1,tau).var()/M+np.minimum(z0,tau).var()/M)
    Nh=2*tau**2*np.log(4/1e-3)/g**2   # two-sided: alpha+beta <= 4 exp(-N g^2/(2tau^2))
    res.append((tau,g,se,Nh)); print(tau,g,se,Nh)
# exact honest mean of min((xp)^2,tau) for Gaussian (needed for two-sided test): numeric integral
def mu0(tau):
    f=lambda a,b: min((a*b)**2,tau)*np.exp(-(a*a+b*b)/(2*V))/(2*np.pi*V)
    L=12*np.sqrt(V); return integrate.dblquad(lambda b,a: f(a,b),-L,L,-L,L,epsabs=1e-7)[0]
best=min(res,key=lambda r:r[3]); print("best",best, "mu0 exact",mu0(best[0]), "mc",np.minimum(z0,best[0]).mean())
json.dump(dict(rows=res,best=best),open("trunc.json","w"),indent=1)
