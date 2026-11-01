# Final Phase 1 integration audit

Verified on 1 November 2026 against the public official repository, after merging the team changes through source revision c135096. The final integration uses the actual checked-out source; no prior successful-run metadata was copied into the official repository.

## Completed checks

- Original ULB/Kaggle CSV: 284,807 observations, 492 frauds and 31 columns. Its checksum and frozen memberships are verified.
- The default full run completed in 584.1 seconds. Runtime/package/configuration details are saved in results/run_metadata.json; status is complete and robustness_completed is true.
- All primary models receive the same 170,528 legitimate training rows. Chronological membership is preserved for 57 validation frauds and 75 test frauds.
- Twenty unit/protocol/integration checks pass. Numerical verification checks eight score arrays, 32 operating points, fractional/top100 capacity, exact threshold serialization, test error groups and configured sensitivity coverage.
- Thirty-three primary GMM candidates and 26 validation-only robustness fits are recorded.
- Seven notebooks execute successfully with no error outputs and valid notebook structure.
- Report files mistakenly uploaded under reports/ieee_report/figures were moved to their canonical report/document paths; the figures folder contains image assets only.
- The IEEEtran report compiles, all rendered pages are checked, and the submission PDF matches the compiled source PDF. The Overleaf archive contains the same source and figure bytes.
- The 15-slide editable presentation passes structural/layout/native chart checks. Every rendered slide is visually checked; its PDF export uses the same renders. No native PowerPoint application test is claimed.
- README, role statuses, scope notes, viva guide and final audit reflect the integrated state rather than the earlier scaffold/pending state.
- Source/configuration hashes are recorded in results/source_manifest.json and checked by scripts/verify_submission.py.
- The repository is public; anonymous URL access and the pushed branch are checked after upload. Raw CSV and serialized joblib files remain excluded from Git.

## Public workflow

GitHub Actions runs the 20 protocol/integration tests and the dataset-free submission artifact checker on pushes and pull requests. The full experiment requires the original Kaggle CSV and is verified locally rather than downloading it in CI. The first remote workflow outcome is checked after pushing.

## Recorded findings

The selected GMM has 20 full-covariance components and reg_covar=1e-6. Test AP is 0.5822 and trapezoidal PR area is 0.5787. At the 0.5% validation FPR policy, TP=57, FP=175, FN=18 and TN=56,711, giving 76% recall, 24.57% precision and 0.308% realized test FPR. All narrative values come from the final run's CSV outputs.

## Limitations and remaining personal steps

The test partition was already inspected, so the official rerun is not a fresh independent holdout. The selected covariance is highly anisotropic, the search is bounded/staged, the diagnostic studies use only three seeds at fixed K=4, and only two historical days are available. Bootstrap intervals omit temporal dependence and model-refitting uncertainty.

Phase 1 implementation and artifacts are integrated; autoencoder/hybrid work remains Phase 2. Each member must still review their work, record genuine personal contribution/review evidence and prepare for the individual viva. This audit does not assert faculty acceptance or independent authorship by a particular person.
