# ============================================================
# Figure 1: Joint Quadrature Density (Single-Shot Scatter)
# Reproducible Colab-ready code for the paper
# "On the Limits of Quantum and Classical Detectors for CV-QKD Attacks"
# ============================================================

# --- 1. Install / Import (Colab already has these) ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# For nicer Colab display
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

np.random.seed(42)   # Reproducible results (change seed for different realizations)

# --- 2. Parameters from the paper ---
V = 2.25          # variance per quadrature (honest & stealth)
C_stealth = 0.95  # off-diagonal correlation for STEALTH attack
N = 8000          # number of quadrature pairs (single-shot samples)
                  # Paper uses ~10^5 for density plots; 8000 is fast & clear for scatter

mean = np.array([0.0, 0.0])

# Covariance matrices
Sigma0 = np.array([[V, 0.0],
                   [0.0, V]])          # Honest (isotropic)

Sigma1 = np.array([[V, C_stealth],
                   [C_stealth, V]])    # Stealth covariance attack

# --- 3. Generate samples ---
samples0 = np.random.multivariate_normal(mean, Sigma0, size=N)
samples1 = np.random.multivariate_normal(mean, Sigma1, size=N)

x0, p0 = samples0[:, 0], samples0[:, 1]
x1, p1 = samples1[:, 0], samples1[:, 1]

# --- 4. Compute empirical statistics (as shown in paper caption) ---
def print_stats(name, x, p):
    var_x = np.var(x, ddof=1)
    var_p = np.var(p, ddof=1)
    C_emp = np.cov(x, p, ddof=1)[0, 1]
    print(f"{name:12s} | Var(x) = {var_x:.2f}   Var(p) = {var_p:.2f}   C = {C_emp:.2f}")

print("Empirical statistics (close to paper caption values):")
print_stats("No-attack",   x0, p0)
print_stats("STEALTH",     x1, p1)

# --- 5. Helper: draw covariance ellipse ---
def plot_cov_ellipse(cov, mean, ax, n_std=2.0, **kwargs):
    """
    Draw an ellipse representing the covariance matrix.
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

    transf = transforms.Affine2D() \
        .rotate_deg(45 if pearson > 0 else -45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# --- 6. Create the figure (exactly like paper style) ---
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)

# Left panel: No-attack
axs[0].scatter(x0, p0, s=0.8, alpha=0.35, color='#1f77b4', rasterized=True)
plot_cov_ellipse(Sigma0, mean, axs[0], n_std=2.0,
                 edgecolor='#1f77b4', linewidth=2.0, fill=False, linestyle='--')
axs[0].set_title('No-attack  ($\Sigma_0$)', fontsize=13, pad=10)
axs[0].set_xlabel('x quadrature')
axs[0].set_ylabel('p quadrature')
axs[0].set_xlim(-7, 7)
axs[0].set_ylim(-7, 7)
axs[0].set_aspect('equal')
axs[0].grid(True, alpha=0.25, linestyle='--')

# Annotation box
text0 = (f"Var(x) = {np.var(x0, ddof=1):.2f}\n"
         f"C = {np.cov(x0, p0, ddof=1)[0,1]:.2f}")
axs[0].text(0.05, 0.95, text0, transform=axs[0].transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# Right panel: Stealth attack
axs[1].scatter(x1, p1, s=0.8, alpha=0.35, color='#d62728', rasterized=True)
plot_cov_ellipse(Sigma1, mean, axs[1], n_std=2.0,
                 edgecolor='#d62728', linewidth=2.0, fill=False, linestyle='--')
axs[1].set_title('Stealth covariance attack  ($\Sigma_1$)', fontsize=13, pad=10)
axs[1].set_xlabel('x quadrature')
axs[1].set_xlim(-7, 7)
axs[1].set_ylim(-7, 7)
axs[1].set_aspect('equal')
axs[1].grid(True, alpha=0.25, linestyle='--')

text1 = (f"Var(x) = {np.var(x1, ddof=1):.2f}\n"
         f"C = {np.cov(x1, p1, ddof=1)[0,1]:.2f}")
axs[1].text(0.05, 0.95, text1, transform=axs[1].transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

fig.suptitle('Figure 1 — Joint quadrature density (single-shot scatter)\n'
             'Scalar-marginal detectors see identical marginals; the off-diagonal C is the only signal',
             fontsize=12, y=1.02)

plt.tight_layout()
plt.show()

# --- 7. Optional: Save high-resolution figure for paper ---
# fig.savefig('figure1_scatter.png', dpi=300, bbox_inches='tight')
print("\n✅ Figure 1 generated successfully!")
print("   (Run this cell in Google Colab — it is fully self-contained)")
