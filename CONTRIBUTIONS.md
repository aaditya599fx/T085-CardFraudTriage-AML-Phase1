# Team T085 contributions

Project 26: Card Fraud Triage | Section E

The roles below follow the team's agreed assignment. Commit evidence records changes submitted through each member's account; it does not establish independent authorship or personal review of every line. The maintainer's implementation, integration and verification used coding-assistant support. Enrollment numbers and email addresses are omitted.

## Phase 1: roles and delivered work

| Team member | Assigned role | Delivered responsibilities and files | Commit evidence |
|---|---|---|---|
| **Gauri Mishra** | Role 1 - Data and exploratory analysis | Dataset/schema checks, chronological split and frozen membership, data provenance, training-only EDA, class counts and figures. Main files: `src/data.py`, `src/eda.py`, `notebooks/01_literature_and_eda.ipynb`, `data/README.md`, `docs/data_card.md`, and related data tables/figures. | `aca10db`, `2798c79`, `aa4babe`, `cd8779b`, `630b413`, `bd0c48b`, `0623b52` |
| **Farhana Pervin** | Role 2 - Feature engineering and preprocessing | Legitimate-only scaling, log-amount transformation, reusable preprocessing/PCA, feature-engineering notebook and preprocessing notes. Main files: `src/features.py`, `notebooks/02_feature_engineering.ipynb`, `docs/preprocessing.md`, `results/models/README.md`. | `6e38170`, `5e652a6`, `5a76d45`, `4d05eac` |
| **Aaditya Yadav** | Role 3 - GMM and classical baselines; integration | Train-only GMM selection and convergence diagnostics; PCA, Isolation Forest and approximate one-class SVM comparators; experiment orchestration, output verification, notebook execution, packaging, public CI and final integration. Main files: `src/gmm_model.py`, `src/baselines.py`, `run_all.py`, integration scripts/tests and baseline notebooks. | `fa84058`, `1230602`, `0d0fb97`, `4587c77`, `7147ea1`, `37da367`, `46d011f` |
| **Bhumi Rastogi** | Role 4 - Metrics and evaluation | AP/PR metrics, validation-calibrated threshold and review-capacity policies, exact threshold reload, evaluation protocol, shared tests, result tables and evaluation figures. Main files: `src/metrics.py`, `src/evaluation.py`, `docs/evaluation_protocol.md`, `tests/test_phase1.py`, `notebooks/05_evaluation.ipynb`, and related tables/figures. | `ecace4f`, `525cc37`, `4c71fa7`, `4820b82`, `fa45689`, `864e286` |
| **Anshvardhan Badkur (Ansh)** | Role 5 - Robustness, error analysis and documentation | Robustness/error-analysis modules, analysis notebook, supporting result artifacts and report/document uploads. Main files: `src/robustness.py`, `src/error_analysis.py`, `notebooks/04_robustness_and_error_analysis.ipynb`, analysis reports and IEEE report assets. | `fb2fe75` (error-analysis module) and related upload history through `c135096`; use file history for individual artifacts. |

## Phase 2: additional recorded contributions

| Team member | Recorded Phase 2 work | Commit / merge evidence |
|---|---|---|
| **Aaditya Yadav** | Initial denoising autoencoder versus frozen-GMM experiment; baseline scores and metrics; initial report and slides; reproducibility and contribution notes; later documentation audit reconciling the hybrid findings. | `728996a`, `e290a5a` |
| **Anshvardhan Badkur (Ansh)** | Expanded comparison at four FPR budgets and matched review capacities; 12 validation-only autoencoder ablations; training curves, metadata and executed comparison notebook; validation-selected hybrid fusion, fusion notebook and tests; updated IEEE report, presentation and Phase 2 CI. | `5f8afa8`, `88bf8bf`; merged in `f00cb5e` and `79e9468` |
| **Gauri Mishra** | Phase 1 data/split contributions support the Phase 2 shared-data protocol. No separate Phase 2 implementation commit is recorded here. | Phase 1 evidence above |
| **Farhana Pervin** | Phase 1 preprocessing contributions support the shared Phase 2 feature transformation. No separate Phase 2 implementation commit is recorded here. | Phase 1 evidence above |
| **Bhumi Rastogi** | Phase 1 model-independent metrics and threshold policies are reused in Phase 2. No separate Phase 2 implementation commit is recorded here. | Phase 1 evidence above |

The later hybrid uses 95% calibrated GMM and 5% calibrated autoencoder scores, selected on mixed-validation AP. Its historical test AP is 0.5754 versus 0.5822 for GMM and 0.0908 for the standalone autoencoder. These are follow-up findings on a test partition already inspected in Phase 1.

## How to inspect the evidence

Use `git show <commit>` for a specific change and `git log -- <file>` for a file's history. Generic upload messages identify submitted artifacts but do not certify independent method development. Existing history has not been rewritten or reassigned among team members.

## Individual review and viva record

Members should add actual reviews, demonstrations or follow-up changes here after completing them. A blank row does not imply a review has occurred.

| Member | Actual review, demonstration or follow-up | Commit / PR evidence | Validation performed | Reviewer |
|---|---|---|---|---|
| | | | | |
