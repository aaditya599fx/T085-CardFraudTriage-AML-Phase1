# Expanded Phase 1 experiment protocol

The first Phase 1 results used an eight-candidate GMM search and were already inspected. This expansion preserves the original chronological membership. It is a follow-up analysis on a previously inspected test partition, not a fresh independent holdout. The test partition was inspected before this official integration run; the current configuration and successful run are identified by the provenance records under results/.

The configuration is fixed in `configs/config.yaml` before the expanded experiment starts and its SHA-256 is saved in `results/experiment_plan.json`.

## Model selection

1. Fit K in {1,2,4,8,12,16,20} with diagonal, full, tied and spherical covariance, using all legitimate training rows, reg_covar=1e-5, n_init=2 and max_iter=300.
2. Select the lowest training BIC among finite, converged candidates.
3. Fix that structural winner. Vary one setting at a time: reg_covar in {1e-6,1e-5,1e-4}, n_init in {1,2,3}, and max_iter in {100,300}. Baseline-equivalent duplicate configurations are not refitted.
4. Select the lowest training BIC across all eligible candidates. Record AIC independently and explain agreement or disagreement.

This is a staged bounded search, not an exhaustive product of structural and optimizer choices. Neither validation AP nor test metrics choose the primary GMM. An optimum at the component-grid boundary is reported as such. Nonconverged fits and numerical failures are logged and excluded from selection.

## Validation calibration and final evaluation

Freeze all models' strict score > threshold policies using legitimate validation scores at FPR budgets {0.1%,0.5%,1%,5%}. Then compute mixed-test metrics. Report AP and trapezoidal PR AUC separately, the base rate, realized FPR, confusion counts and review workload. Capacity policies use top {0.1%,0.5%,1%} of each scored batch and top 100, with stable chronological tie breaking. They use no labels to choose queue membership.

## Robustness studies

Use only legitimate training and mixed validation frames. No test frame is passed to the robustness function.

- Cross input dimensions {5,10,20,29} with all four covariance types at fixed K=4, seed=26, n_init=2, max_iter=200 and reg_covar=1e-5.
- Repeat each covariance at full dimension with seeds {13,26,42}. Report validation AP mean/SD, pairwise score-rank correlations and per-observation score dispersion.
- Fit full-covariance K=4 on nested seeded {10%,50%,100%} normal-training subsets. Keep full-training preprocessing fixed to isolate the GMM sample-size effect.
- Log convergence, numerical warnings, component weights and covariance eigenvalue diagnostics.

AIC/BIC comparisons are meaningful only for equal observation sets and equal feature spaces. Do not rank PCA dimensions or training fractions by their raw information criteria. These are diagnostic studies and do not replace the primary GMM using validation or test AP.

## Post-freeze error analysis

Inspect GMM false negatives and false positives on test at the predetermined 0.5% validation FPR budget. Save missed-fraud rows, top 20 false-positive rows, group amount/time/score summaries and standardized feature deviations. These are descriptive findings. They do not feed back into fitting or tuning.

## Phase 2 freeze

Reuse `split_indices.npz`, the full preprocessing pipeline, feature order, and the model-independent evaluation harness. Neural model training remains Phase 2. Disclose reuse of the already inspected test set in future comparisons.
