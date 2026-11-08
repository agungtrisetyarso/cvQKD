"""Numerical checks backing the revised theorems (v2)."""
import numpy as np, json, itertools
from scipy.special import ive, gammaln, logsumexp
from scipy import integrate
from keyrate import keyrate, c_max, state, cond_het, S, I2
OUT = {}

# 1. Lancaster: chi^2 of Kibble bivariate gamma vs independence = rho^2/(1-rho^2), any shape a
def kibble_chi2(a, rho, M=1400):
    sd = np.sqrt(a); lo = max(1e-9, a - 12 * sd); hi = a + 14 * sd + 30
    x = np.linspace(lo, hi, M); dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    z = 2 * np.sqrt(rho * X * Y) / (1 - rho)
    logf = ((a - 1) / 2) * np.log(X * Y / rho) - (X + Y) / (1 - rho) - gammaln(a) - np.log(1 - rho) \
        + np.log(ive(a - 1, z)) + z
    logm = (a - 1) * np.log(x) - x - gammaln(a)
    logL = logf - logm[None, :] - logm[:, None]
    mass = np.exp(logf).sum() * dx * dx
    chi2 = (np.exp(logf + logL)).sum() * dx * dx - 1
    return chi2, mass
rho = 0.42**2
OUT["lancaster"] = []
for a in (1.5, 5, 50, 500):
    c2, m = kibble_chi2(a, rho)
    OUT["lancaster"].append((a, c2, rho**2 / (1 - rho**2), m)); print("Kibble a=", a, "chi2=", c2, "theory", rho**2/(1-rho**2), "mass", m)

def marg_tv_bound(r):
    chi2 = 1 / ((1 - r**2) * (1 - r**4)) - 1
    return chi2, 0.5 * np.sqrt(chi2)
OUT["chi2_r042"] = marg_tv_bound(0.42222)
print("chi2 bound r=.422", OUT["chi2_r042"])

# 2. multimode CP budget: single entry kappa between quadratures i,j of a 2-mode channel
def lam_min(n_, T, K):
    Om = np.kron(np.eye(2), np.array([[0, 1], [-1, 0]]))
    H = n_ * np.eye(4) + K + 1j * (1 - T) * Om
    return np.linalg.eigvalsh(H).min()
T, xi = 0.5, 0.02; n_ = 1 - T + T * xi
budgets = {}
for name, (i, j) in {"x1p1": (0, 1), "x1x2": (0, 2), "x1p2": (0, 3)}.items():
    lo, hi = 0, n_
    for _ in range(60):
        mid = (lo + hi) / 2
        K = np.zeros((4, 4)); K[i, j] = K[j, i] = mid
        lo, hi = (mid, hi) if lam_min(n_, T, K) >= -1e-14 else (lo, mid)
    budgets[name] = lo
print("multimode single-entry budgets", budgets, "closed form 1-mode", c_max(T, xi), "T xi", T*xi)
OUT["mm_budget"] = budgets
# sufficient condition check: ||K||op <= c_max for random K with zero diag blocks?
rng = np.random.default_rng(1); worst = 1e9
for _ in range(20000):
    A = rng.standard_normal((4, 4)); K = (A + A.T); np.fill_diagonal(K, 0)
    K *= c_max(T, xi) / np.abs(np.linalg.eigvalsh(K)).max() * 0.999
    worst = min(worst, lam_min(n_, T, K))
print("random K with ||K||<=cmax: min eig", worst); OUT["mm_opnorm_suff_min_eig"] = worst

# 3. key-rate safety sweep (heterodyne), both signs of c
grid = list(itertools.product([0.9, 0.5, 0.1, 0.01], [0.01, 0.05, 0.1], [2.0, 4.0, 10.0], [(0.6, 0.05), (1 - 1e-9, 0.0)]))
mind = 1e9; argm = None; cnt = 0
for T_, xi_, VA_, (eta_, vel_) in grid:
    K0 = keyrate(VA_, T_, xi_, 0.0, eta_, vel_)[0]
    cm = c_max(T_, xi_)
    for f in (-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.999):
        d = keyrate(VA_, T_, xi_, f * cm, eta_, vel_)[0] - K0; cnt += 1
        if d < mind: mind, argm = d, (T_, xi_, VA_, eta_, vel_, f)
print("keyrate sweep: points", cnt, "min K(c)-K(0)", mind, argm)
OUT["sweep"] = dict(points=cnt, min_diff=mind, at=argm)
# chi monotone check separately
# 4. Thm 6 premise: KL between Gaussian and Gamma-volatility mixture vs dk^2
V = 2.25
def kl_gamma(eps):
    k = 1 / eps
    yq, wq = np.polynomial.laguerre.laggauss(200)       # for Gamma via substitution
    # Gamma(k, V/k) quadrature using generalized Laguerre (alpha=k-1)
    from scipy.special import roots_genlaguerre
    t, w = roots_genlaguerre(300, k - 1)
    y = t * V / k; logw = np.log(w) - gammaln(k)
    R = np.linspace(1e-8, 2 * V * 60, 60000)  # R = x^2+p^2 ; under H0 R ~ Exp(mean 2V)
    logp1 = logsumexp(logw[None, :] - np.log(2 * np.pi * y[None, :]) - R[:, None] / (2 * y[None, :]), axis=1)
    logp0 = -np.log(2 * np.pi * V) - R / (2 * V)
    dens0 = np.exp(-R / (2 * V)) / (2 * V)
    return np.trapezoid(dens0 * (logp0 - logp1), R)
OUT["kl_gamma"] = []
for eps in (0.02, 0.05, 0.1, 0.2, 0.4, 0.674):
    D = kl_gamma(eps); dk = V**2 * eps
    OUT["kl_gamma"].append((eps, dk, D, D / dk**2)); print("eps", eps, "dk", dk, "D", D, "D/dk^2", D / dk**2)

# 5. truncated energy detector: mean gap and Hoeffding constant for Gamma attack (k=1.484)
k = 1 / 0.674; rng = np.random.default_rng(3); M = 4_000_000
x0 = rng.standard_normal((M, 2)) * np.sqrt(V)
y1 = rng.gamma(k, V / k, size=M); x1 = rng.standard_normal((M, 2)) * np.sqrt(y1)[:, None]
z0 = (x0[:, 0] * x0[:, 1])**2; z1 = (x1[:, 0] * x1[:, 1])**2
best = None
for tau in (5, 10, 20, 30, 40, 60, 80, 120):
    g = np.minimum(z1, tau).mean() - np.minimum(z0, tau).mean()
    Nh = 2 * tau**2 * np.log(2 / 1e-3) / g**2      # alpha+beta <= 2 exp(-N g^2/(2 tau^2))
    if best is None or Nh < best[2]: best = (tau, g, Nh)
    print("tau", tau, "gap", g, "N(Hoeffding, a+b<=1e-3)", Nh)
OUT["trunc_best"] = best
json.dump(OUT, open("theory_checks.json", "w"), indent=1, default=float)
