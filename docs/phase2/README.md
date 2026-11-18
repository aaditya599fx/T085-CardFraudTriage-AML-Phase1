# Phase 2 reproducibility

Run `python run_phase2.py` from this repository with the original ULB/Kaggle `data/raw/creditcard.csv`. It verifies the frozen Phase 1 split, trains an undercomplete denoising neural autoencoder on legitimate transactions, reuses frozen GMM scores, and writes metrics, thresholds, score arrays and the PR curve to `results/phase2/`. The dataset and fitted joblib objects are not committed.

The IEEE-style report source and compiled PDF are in `reports/phase2_ieee/`; the editable presentation and PDF are in this folder. The test partition had already been inspected in Phase 1. The current neural optimizer reached its 35-epoch cap; this is a measured comparison, not a claim of universal neural performance.
