# Phase 1 completion and scope

Phase 1 contains classical machine learning. Autoencoder training and hybrid innovation remain Phase 2.

| Requirement | Implementation/evidence |
|---|---|
| Data quality and training EDA | src/data.py, src/eda.py, notebook 01, data card |
| Explicit reusable split membership | data/processed/split_indices.npz plus provenance/checksum validation |
| Leakage-safe preprocessing | src/features.py, notebook 02, saved local pipelines |
| GMM selection and fitting sensitivity | 28 structural candidates plus staged optimizer trials; notebook 03 |
| Model-agnostic evaluation | evaluate_scores, frozen validation thresholds, top-100 and fractional review policies |
| Validation and test PR curves | results/figures/pr_curves.png; notebook 05 |
| Confusion matrices and workload | gmm_confusion_matrices.png, operating_points.csv, review_workload.csv |
| Dimensionality/covariance/seed/sample sensitivity | src/robustness.py, notebook 04, robustness tables |
| Missed-fraud and false-positive analysis | src/error_analysis.py, error-analysis tables and narrative |
| Final assembled narrative | notebook 06, README, IEEE report, presentation and viva guide |
| Individual responsibilities | CONTRIBUTIONS.md with the team's supplied role assignments |

A successful default `python run_all.py` executes the complete experiment. `--skip-robustness` is a quick rerun and is not a complete reproduction of the submission.

The original mixed-test set was already inspected before the expansion. The report discloses test reuse. Train-only model selection and validation-only robustness help preserve the protocol, but do not create a new independent holdout.

Before submission, each member should review their assigned work, record actual contribution evidence, prepare for individual viva and verify anonymous access to the public official repository. Those personal/administrative steps cannot be certified by automated execution.

Phase 2 must reuse the frozen split membership, preprocessing, feature order and evaluation harness. Predetermine neural choices using validation and disclose test reuse. A hybrid needs an appropriate rationale and ablation evidence.

Final automated checks also include `python scripts/verify_outputs.py` for numerical contracts and `python scripts/verify_submission.py` for artifact paths, notebook execution, source hashes and matching report/archive versions.
