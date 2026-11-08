"""Gaussian key rates (reverse reconciliation, trusted detector noise) and
the PE-consistent stealth budget for phase-sensitive (cross-quadrature) noise.
Shot-noise units: vacuum covariance = I."""
import numpy as np

I2 = np.eye(2); Z2 = np.diag([1., -1.]); X2 = np.array([[0., 1.], [1., 0.]])
OM1 = np.array([[0., 1.], [-1., 0.]])

def omega(n):
    return np.kron(np.eye(n), OM1)

def g(nu):
    nu = np.asarray(nu, float)
    out = np.zeros_like(nu)
    m = nu > 1 + 1e-12
    a, b = (nu[m] + 1) / 2, (nu[m] - 1) / 2
    out[m] = a * np.log2(a) - b * np.log2(b)
    return out

def S(gam):
    n = gam.shape[0] // 2
    ev = np.abs(np.linalg.eigvals(1j * omega(n) @ gam))
    nus = np.sort(ev)[::2]
    return float(np.sum(g(nus)))

def bs(eta):
    t, r = np.sqrt(eta), np.sqrt(1 - eta)
    return np.block([[t * I2, r * I2], [-r * I2, t * I2]])

def cond_het(gam, keep, meas):
    """Covariance of modes `keep` after heterodyne of mode `meas`."""
    idx_k = np.concatenate([[2 * k, 2 * k + 1] for k in keep])
    idx_m = [2 * meas, 2 * meas + 1]
    A = gam[np.ix_(idx_k, idx_k)]; B = gam[np.ix_(idx_m, idx_m)]
    Cc = gam[np.ix_(idx_k, idx_m)]
    return A - Cc @ np.linalg.inv(B + I2) @ Cc.T

def state(VA, T, xi, c, eta, vel, het=True):
    """Modes: 0=A, 1=B(after detector), 2=F, 3=G. Channel output noise
    Y = (1-T+T xi) I + c X.  Convention (v2.1): v_el is the electronic noise of
    EACH homodyne arm of the receiver (Lodewyck/Fossier); a heterodyne receiver
    then adds 2 v_el referred to Bob's mode, i.e. chi_het=(2-eta+2v_el)/eta.
    The equivalent single-mode model is an EPR source of variance
    v = 1 + m v_el/(1-eta), m = 2 (heterodyne) or 1 (homodyne)."""
    V = VA + 1
    s = np.sqrt(V**2 - 1)
    gA = V * I2
    gB = T * V * I2 + (1 - T + T * xi) * I2 + c * X2
    sAB = np.sqrt(T) * s * Z2
    v = 1 + (2.0 if het else 1.0) * vel / (1 - eta)
    sv = np.sqrt(v**2 - 1)
    gam = np.zeros((8, 8))
    gam[0:2, 0:2] = gA; gam[2:4, 2:4] = gB
    gam[0:2, 2:4] = sAB; gam[2:4, 0:2] = sAB.T
    gam[4:6, 4:6] = v * I2; gam[6:8, 6:8] = v * I2
    gam[4:6, 6:8] = sv * Z2; gam[6:8, 4:6] = sv * Z2
    Sbs = np.eye(8); Sbs[2:6, 2:6] = bs(eta)
    gAB_ch = gam[0:4, 0:4].copy()          # A + channel output (E purifies)
    gam = Sbs @ gam @ Sbs.T
    return gam, gAB_ch

def keyrate(VA, T, xi, c=0.0, eta=0.6, vel=0.05, beta=0.95):
    gam, gAB = state(VA, T, xi, c, eta, vel)
    gBp = gam[2:4, 2:4]
    gB_A = cond_het(gam, [1], 0)                       # Alice heterodynes
    I_AB = 0.5 * np.log2(np.linalg.det(gBp + I2) / np.linalg.det(gB_A + I2))
    S_AB = S(gAB)
    S_cond = S(cond_het(gam, [0, 2, 3], 1))
    chi = S_AB - S_cond
    return beta * I_AB - chi, I_AB, chi

def c_max(T, xi):
    """Largest channel cross-quadrature noise c with Y + i(1-T)Omega >= 0."""
    return np.sqrt(max((1 - T + T * xi)**2 - (1 - T)**2, 0.0))

if __name__ == "__main__":
    VA, T, eta, vel = 4.0, 0.5, 0.6, 0.05
    for xi in (0.02, 0.05):
        K, Iab, chi = keyrate(VA, T, xi)
        print(f"xi={xi}: K={K:.4f} I={Iab:.4f} chi={chi:.4f}")
    cm = c_max(T, 0.02)
    print("c_max(channel)=", cm, " Bob-side C_max=eta*c_max=", eta * cm)
    for f in (0, 0.25, 0.5, 0.75, 0.999):
        K, Iab, chi = keyrate(VA, T, 0.02, c=f * cm)
        print(f"c={f*cm:.4f}: K={K:.5f} I={Iab:.5f} chi={chi:.5f}")
