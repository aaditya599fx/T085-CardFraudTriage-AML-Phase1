# Robustness and stability analysis

26 predetermined trials use legitimate training data and mixed validation only. Test observations never enter this module. K=4, n_init=2, max_iter=200 and reg_covar=1e-5 are fixed for the diagnostic studies.

## Dimensionality and covariance

Cross dimensions 5,10,20,29 with diagonal, full, tied and spherical covariance. PCA and StandardScaler fit on legitimate training only. Full covariance validation AP across dimensions is 5d: 0.0211, 10d: 0.0118, 20d: 0.0085, 29d: 0.0664.

Maximum covariance condition numbers, minimum covariance eigenvalues, component weights, warnings and convergence are recorded per trial. 0 of 26 robustness trials did not converge or failed numerically. A covariance condition number above 1e6 is flagged diagnostically; the cutoff is not a proof of invalidity. Spherical covariance has condition number one within each component by construction, which does not establish good ranking or comparable variance scales between components.

Changing dimension affects both retained information and conditioning. A drop in condition number cannot by itself prove a better fraud detector. Raw AIC/BIC must not rank these different feature spaces.

## Multiple seeds

All four covariance types at full dimension are repeated with seeds 13,26,42.

| Covariance | Runs | Mean validation AP | Sample SD AP | Maximum condition |
|---|---:|---:|---:|---:|
| diag | 3 | 0.0589 | 0.0002 | 353.2 |
| full | 3 | 0.3534 | 0.2488 | 332,244.9 |
| spherical | 3 | 0.0298 | 0.0016 | 1.0 |
| tied | 3 | 0.1075 | 0.0581 | 51,095.5 |

Pairwise validation score-rank correlations range from 0.7170 to 1.0000. `seed_score_dispersion.csv.gz` stores per-observation mean and sample SD of raw negative log density across seeds. Raw-score SD depends on score scale; rank correlations help interpret ordering stability.

## Training size

Nested seeded 10%,50%,100% subsets isolate GMM fitting-population size at 29 dimensions with full covariance. The full legitimate-training preprocessing is held fixed, so this is a GMM sample-size experiment, not a full-pipeline learning curve.

| Fraction | Fit rows | Validation AP | Maximum condition |
|---|---:|---:|---:|
| 10% | 17,052 | 0.0535 | 949,093.3 |
| 50% | 85,264 | 0.6208 | 363,643.8 |
| 100% | 170,528 | 0.0664 | 259,408.6 |

## Interpretation limits

These are descriptive sensitivity experiments, not a statistical guarantee of stability. Some covariance assumptions constrain condition numbers by construction. Validation has only 57 frauds, seeds are only three, and K=4 is fixed here. The primary model uses a separate train-only structural/optimizer selection. Do not choose a winner across dimensions from the raw density criteria or use test outcomes to tune these experiments.
