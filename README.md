# cvQKD

**Reproducible simulation code** for the paper:

> **On the Limits of Quantum and Classical Detectors for Continuous-Variable QKD Attacks**  
> Agung Trisetyarso  
> *IEEE Transactions on Machine Learning in Communications and Networking* (submitted, 2026)

This repository contains clean, self-contained Python scripts that exactly reproduce the key numerical results, Table I, and all main figures of the paper. All scripts are designed to run instantly in **Google Colab**.

## What This Repository Demonstrates

The code provides a rigorous, reproducible verification of the paper’s central theoretical results:

- **Lemma 1 (Marginal Blindness)**: A physically admissible covariance-only (“stealth”) attack renders every scalar-marginal detector information-theoretically blind (AUC ≈ ½), while a covariance detector detects it with perfect accuracy.
- **Covariance Sufficiency (Proposition 1 & Corollary 1)**: Within the Gaussian attack family, the classical sample covariance is a sufficient statistic. No quantum or quantum-inspired model can outperform it.
- Phase-space visualizations and Glauber-Sudarshan P-function approximations for practical CV-QKD attack models.

## Repository Contents

| File                        | Reproduces          | Description |
|----------------------------|---------------------|-----------|
| `quad.py`                  | **Figure 1**        | Single-shot scatter plots of joint quadratures (x, p). Shows identical marginals but clearly different correlation for the stealth attack. |
| `density.py`               | **Figure 2**        | Binned joint quadrature density (2×10⁵ samples) with 1σ/2σ covariance ellipses. |
| `Glauber_Sudarshan.py`     | **Figure 3**        | Glauber-Sudarshan P-function approximations (histogram + KDE) for six attack models (BA, CA, HA, LA, NA/honest, SA). |
| `roc.py`                   | ROC curves          | Full end-to-end simulation of scalar-marginal, covariance, and higher-order detectors on the stealth attack. Computes and plots ROC + AUC (demonstrates Lemma 1). |
| `auc.py`                   | **Table I**         | Grouped bar plot of detection AUC values for STEALTH and COVTILT attacks across the three detector families. |
| `roc1.py`                  | Variant ROC         | Alternative/extended ROC visualization script. |

## Key Results Reproduced

| Attack vs. No-attack | Scalar-marginal | Covariance | Higher-order |
|----------------------|------------------|------------|--------------|
| **STEALTH** (matched V, C ≠ 0) | ~0.55 (blind)   | **1.000**  | ~0.57        |
| **COVTILT** (overt)            | **1.000**       | **1.000**  | ~0.53        |

These numbers match the theoretical predictions of the paper.

## Quick Start (Recommended)

### Option 1: Google Colab (easiest)

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook → upload any `.py` file from this repo, **or**
3. Create a new notebook and paste the entire content of the desired script.
4. Run all cells.

All dependencies are pre-installed in Colab.

### Option 2: Local execution

```bash
git clone https://github.com/agungtrisetyarso/cvQKD.git
cd cvQKD

python quad.py                    # Figure 1
python density.py                 # Figure 2
python Glauber_Sudarshan.py       # Figure 3
python roc.py                     # ROC curves + AUC
python auc.py                     # Table I bar plot
