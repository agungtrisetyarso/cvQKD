# ============================================================
# Figure 5: Detection Performance Summary (AUC Comparison)
# Reproducible Google Colab code
# Based on Table I from the manuscript
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Colab-friendly settings
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# ================== Data from Table I (exactly as in the paper) ==================
data = {
    'Attack': ['STEALTH (matched V, C ≠ 0)', 'STEALTH (matched V, C ≠ 0)', 'STEALTH (matched V, C ≠ 0)',
               'COVTILT (overt)', 'COVTILT (overt)', 'COVTILT (overt)'],
    'Detector': ['Scalar-marginal', 'Covariance', 'Higher-order',
                 'Scalar-marginal', 'Covariance', 'Higher-order'],
    'AUC': [0.553, 1.000, 0.570,
            1.000, 1.000, 0.527]
}

df = pd.DataFrame(data)

# ================== Create Figure 5 ==================
fig, ax = plt.subplots(figsize=(11, 6))

# Grouped bar plot
sns.barplot(
    data=df,
    x='Attack',
    y='AUC',
    hue='Detector',
    palette=['#1f77b4', '#d62728', '#2ca02c'],  # Blue, Red, Green
    ax=ax,
    edgecolor='black',
    linewidth=0.8
)

# Add value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10, fontweight='bold')

# Styling
ax.set_ylim(0, 1.15)
ax.set_ylabel('Area Under ROC Curve (AUC)', fontsize=12)
ax.set_xlabel('')
ax.set_title(
    'Figure 5 — Detection Performance of Different Detector Families\n'
    '(Five-fold cross-validated AUC on simulated CV-QKD data)\n'
    'STEALTH attack is invisible to scalar-marginal detectors but perfectly detected by covariance',
    fontsize=13, pad=15
)

# Add horizontal line at chance level (0.5)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance level (AUC = 0.5)')

# Legend
ax.legend(title='Detector Family', loc='upper right', framealpha=0.95)

# Add annotation box explaining the key result
textstr = (
    'Key Result (Lemma 1):\n'
    '• STEALTH attack preserves all marginals\n'
    '• Scalar-marginal detector → AUC ≈ 0.5 (blind)\n'
    '• Covariance detector → AUC = 1.0 (perfect)\n'
    '• Higher-order detector → near chance'
)
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# ================== Optional: Save figure ==================
# fig.savefig('figure5_auc_comparison.png', dpi=300, bbox_inches='tight')

print("✅ Figure 5 generated successfully!")
print("   - Grouped bar plot of AUC values from Table I")
print("   - Clearly shows the sufficiency boundary (Lemma 1)")
print("   - Ready for paper or presentation use")
print("   - Fully self-contained for Google Colab")
