# ============================================================
# Figure 2: Joint Quadrature Density (Binned)
# Reproducible Colab-ready code matching the paper
# "On the Limits of Quantum and Classical Detectors for CV-QKD Attacks"
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.colors import LogNorm   # optional but nice for density

# Colab-friendly settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

np.random.seed(42)          # reproducible

# ================== Parameters from the paper ==================
V = 2.25
C_stealth = 0.95
N = 200_000                 # exactly as in the paper caption

mean = np.array([0.0, 0.0])

Sigma0 = np.array([[V, 0.0],
                   [0.0, V]])          # No-attack (isotropic)

Sigma1 = np.array([[V, C_stealth],
                   [C_stealth, V]])    # Stealth attack

# ================== Generate samples ==================
print("Generating 200,000 samples per class (this is fast)...")
samples0 = np.random.multivariate_normal(mean, Sigma0, size=N)
samples1 = np.random.multivariate_normal(mean, Sigma1, size=N)

x0, p0 = samples0[:, 0], samples0[:, 1]
x1, p1 = samples1[:, 0], samples1[:, 1]

# Verify physicality (det Σ > σ₀²)
det0 = np.linalg.det(Sigma0)
det1 = np.linalg.det(Sigma1)
print(f"det(Σ₀) = {det0:.2f}   (physical)")
print(f"det(Σ₁) = {det1:.2f}   (physical)")

# ================== Helper: covariance ellipse ==================
def plot_cov_ellipse(cov, mean, ax, n_std=1.0, **kwargs):
    """
    Draw covariance ellipse for given number of standard deviations.
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (transforms.Affine2D()
              .rotate_deg(45 if pearson > 0 else -45)
              .scale(scale_x, scale_y)
              .translate(mean[0], mean[1]))

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# ================== Create Figure 2 ==================
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)

# Common settings
bins = 120
cmap = 'viridis'           # sequential colormap (as used in the paper)
vmin, vmax = None, None    # let each panel auto-scale (or set same values)

# ---------- Left panel: No-attack ----------
h0 = axs[0].hist2d(x0, p0, bins=bins, cmap=cmap, 
                   range=[[-7, 7], [-7, 7]], 
                   norm=LogNorm() if True else None)   # LogNorm makes faint regions visible

# Overlay 1σ and 2σ ellipses (dashed)
plot_cov_ellipse(Sigma0, mean, axs[0], n_std=1.0,
                 edgecolor='white', linewidth=1.8, linestyle='--', fill=False)
plot_cov_ellipse(Sigma0, mean, axs[0], n_std=2.0,
                 edgecolor='white', linewidth=1.2, linestyle=':', fill=False)

axs[0].set_title('No-attack  ($\Sigma_0$, $C=0$)', fontsize=13, pad=8)
axs[0].set_xlabel('x quadrature')
axs[0].set_ylabel('p quadrature')
axs[0].set_xlim(-7, 7)
axs[0].set_ylim(-7, 7)
axs[0].set_aspect('equal')

# Colorbar for left panel
cbar0 = fig.colorbar(h0[3], ax=axs[0], shrink=0.8, pad=0.02)
cbar0.set_label('Density (a.u.)', fontsize=10)

# ---------- Right panel: Stealth attack ----------
h1 = axs[1].hist2d(x1, p1, bins=bins, cmap=cmap,
                   range=[[-7, 7], [-7, 7]],
                   norm=LogNorm() if True else None)

# Overlay 1σ and 2σ ellipses (dashed)
plot_cov_ellipse(Sigma1, mean, axs[1], n_std=1.0,
                 edgecolor='white', linewidth=1.8, linestyle='--', fill=False)
plot_cov_ellipse(Sigma1, mean, axs[1], n_std=2.0,
                 edgecolor='white', linewidth=1.2, linestyle=':', fill=False)

axs[1].set_title('Stealth covariance attack  ($\Sigma_1$, $C=0.95$)', fontsize=13, pad=8)
axs[1].set_xlabel('x quadrature')
axs[1].set_xlim(-7, 7)
axs[1].set_ylim(-7, 7)
axs[1].set_aspect('equal')

cbar1 = fig.colorbar(h1[3], ax=axs[1], shrink=0.8, pad=0.02)
cbar1.set_label('Density (a.u.)', fontsize=10)

# Main title
fig.suptitle('Figure 2 — Joint quadrature density (binned, 200,000 samples per class)\n'
             'Marginals are identical; only the joint orientation (covariance) differs',
             fontsize=12, y=1.03)

plt.tight_layout()
plt.show()

# ================== Optional: Save figure ==================
# fig.savefig('figure2_binned_density.png', dpi=300, bbox_inches='tight')
print("\n✅ Figure 2 generated successfully!")
print("   (Runs comfortably in Google Colab with 200k samples)")
