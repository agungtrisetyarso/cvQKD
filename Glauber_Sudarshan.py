# ============================================================
# Figure 3: Glauber-Sudarshan P-function Approximations
# under Different Attack Models in CV-QKD
# Fully reproducible Google Colab code
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Colab display settings
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

np.random.seed(42)

# ================== Simulation Parameters ==================
N_samples = 50_000          # samples per attack type (fast in Colab)
V = 2.25                    # reference variance (honest channel)

# Attack labels exactly as in the paper
attack_labels = [
    '(a) BA attack',
    '(b) CA attack',
    '(c) HA attack',
    '(d) LA attack',
    '(e) NA attack (honest channel)',
    '(f) SA attack'
]

# Different statistical parameters for each attack type
# (These are representative models that produce visibly different P-functions)
attack_params = [
    {'mean': 0.0,   'std': 1.1},     # BA attack - slightly squeezed / lower variance
    {'mean': 0.3,   'std': 1.6},     # CA attack - displaced + higher noise
    {'mean': -0.2,  'std': 1.4},     # HA attack - small displacement
    {'mean': 0.0,   'std': 2.8},     # LA attack - high variance / noisy
    {'mean': 0.0,   'std': np.sqrt(V)},  # NA - honest isotropic (reference)
    {'mean': 0.8,   'std': 1.3},     # SA attack - strong displacement
]

# ================== Generate data for each attack ==================
data_dict = {}
for i, params in enumerate(attack_params):
    data = np.random.normal(params['mean'], params['std'], N_samples)
    data_dict[i] = data

# ================== Create Figure 3 (2 rows × 3 columns) ==================
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
axes = axes.flatten()

x_grid = np.linspace(-8, 8, 600)   # common grid for smooth curves

for idx, ax in enumerate(axes):
    data = data_dict[idx]
    label = attack_labels[idx]
    params = attack_params[idx]

    # 1. Yellow histogram (binned approximation)
    ax.hist(data, bins=90, density=True, color='#FFD700', alpha=0.75,
            edgecolor='none', label='Binned P-approx')

    # 2. Red smoothed curve (Gaussian KDE)
    kde = gaussian_kde(data)
    density = kde(x_grid)
    ax.plot(x_grid, density, color='#D62728', linewidth=2.5,
            label='Smoothed P-representation')

    # Styling
    ax.set_title(label, fontsize=11, pad=6)
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel('Quadrature value')
    ax.set_ylabel('Probability density')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Add small text with mean/std for reference
    ax.text(0.02, 0.98, f"μ={params['mean']:.1f}, σ={params['std']:.2f}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

# Main title
fig.suptitle('Figure 3 — Glauber-Sudarshan P-function approximations\n'
             'under different attack models in CV-QKD\n'
             '(Yellow = binned histogram from quadrature samples, Red = smoothed P-representation)',
             fontsize=13, y=1.02)

plt.tight_layout()
plt.show()

# ================== Optional: Save high-resolution figure ==================
# fig.savefig('figure3_pfunction_attacks.png', dpi=300, bbox_inches='tight')

print("✅ Figure 3 generated successfully!")
print("   - 6 panels with histogram + KDE curve")
print("   - Matches the style and layout of the paper")
print("   - Fully self-contained for Google Colab")
