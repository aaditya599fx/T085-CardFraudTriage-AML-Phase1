# Card Fraud Triage: GMM Density Scoring vs Autoencoder Anomaly Detector

**Team T085 · Section E · Project 26 · Phase 1: Machine Learning**

The project learns legitimate transaction behavior and ranks later mixed transactions for a limited fraud review queue. Phase 1 includes GMM density scoring, classical comparators, numerical sensitivity studies and descriptive error analysis. Autoencoder training and hybrid evaluation are Phase 2.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Download the original ULB/Kaggle creditcard.csv into data/raw/
python run_all.py
python -m unittest discover -s tests -v
python scripts/verify_outputs.py
python scripts/verify_submission.py
jupyter lab
```

Dataset: [ULB/Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Original CSV: 284,807 rows, 492 frauds, no missing values. The CSV and fitted joblib models are excluded from Git. `results/models/README.md` documents regeneration. An external CSV path works with `python run_all.py --data /absolute/path/creditcard.csv`.

The default command executes the full project. The final integration was also run on the original dataset in this official repository; current provenance is saved under `results/`. `--skip-robustness` is only a quick rerun. Saved split indices are reused and checked against the dataset checksum, configured fractions, chronology and legitimate training membership. Mismatches fail instead of silently changing the experiment.

## Team responsibilities

| Role | Responsibility |
|---|---|
| 1 | Data, training EDA and saved chronological splits |
| 2 | Preprocessing, feature decisions and PCA |
| 3 | GMM selection and integration |
| 4 | Metrics, threshold calibration and review capacity |
| 5 | Robustness, error analysis and final narrative |

Assignments are ownership and review responsibilities. Actual personal contributions require real commit/review evidence in `CONTRIBUTIONS.md`.

## Repository structure

```text
configs/                 Fixed experiment configuration and team metadata
src/                     Data, preprocessing, GMM, baselines, metrics, robustness, error analysis
tests/                   20 protocol, model and integration checks
notebooks/               Seven executed notebooks, including an assembled Phase 1 summary
data/processed/          Frozen split indices, dataset checksum and split manifest
results/figures/         EDA, model selection, validation/test PR, sensitivity and error plots
results/tables/          Selection, triage, robustness, workload and error-analysis results
results/scores/          Row-aligned validation/test score arrays
results/models/          Model regeneration guide; fitted binaries remain local
docs/                    Protocols, data card, audit, editable presentation and PDF
reports/                 Literature, robustness, error analysis and compiled IEEE report
reports/ieee_report/     Compilable IEEEtran source and referenced figures
scripts/                 Reproduction, verification and report/presentation rebuilding
guides/                  Phase 1 checklist and viva preparation
run_all.py               Complete reproducible experiment
```

## Experimental protocol

- Approximate chronological 60/20/20 partitions, preserving equal timestamps. There are 170,528 legitimate training records, 57 validation frauds and 75 test frauds.
- Explicit `data/processed/split_indices.npz` fixes membership for every model and Phase 2. Provider-created PCA features cannot be independently refit.
- Inputs are V1-V28 and log1p(Amount). Time defines the split and EDA. Preprocessing fits on legitimate training only.
- Structural search crosses K={1,2,4,8,12,16,20} and four covariance types. Predetermined one-factor trials vary regularization, initialization count and iteration limit on the structural winner. Minimum converged training BIC chooses the final GMM. AIC is reported separately.
- The same training rows feed PCA reconstruction, Isolation Forest and an explicitly approximate RBF one-class SVM. Baseline settings stay fixed.
- Validation calibrates strict score>threshold policies at FPR budgets 0.1%, 0.5%, 1% and 5%. Capacity policies review the top 0.1%, 0.5%, 1% or top 100 scores, with stable chronological tie breaking.
- Robustness uses validation only: dimensions 5/10/20/29, all four covariance types, three seeds and nested 10%/50%/100% training subsets at fixed K=4. Raw AIC/BIC are not compared across dimensional spaces or different fitting populations.
- Post-freeze error analysis describes missed frauds, top false positives and review workload. It does not retune the model.

**Test reuse disclosure:** this is an expanded follow-up to the original Phase 1 run. The same test partition was already inspected. Model selection still uses training BIC, robustness uses validation, and thresholds freeze before the current test computation, but these results are not a fresh independent holdout. Earlier exploratory results are not treated as a new independent test set; the official run provenance is recorded in `results/experiment_plan.json` and `results/run_metadata.json`.

## Recorded results

Training BIC selected **20 full-covariance components**, reg_covar=1e-06, n_init=2 and max_iter=300. 33 primary fits and 26 robustness fits are recorded. The selected covariance condition number is approximately 17,759,113.

| Model | Test AP | Trapezoidal PR AUC |
|---|---:|---:|
| gmm | 0.5822 | 0.5787 |
| isolation_forest | 0.0538 | 0.0524 |
| pca | 0.1573 | 0.1540 |
| approx_ocsvm | 0.0014 | 0.0014 |

GMM AP's descriptive 95% stratified bootstrap interval is 0.4634-0.7123. At the **0.5% validation FPR budget**, GMM catches **57/75 frauds**, with **175 false positives**, **0.308% actual test FPR**, and **24.57% precision** across **232 alerts**.

The separate top-0.5% capacity policy reviews 284 records and catches 57 frauds, precision 20.07% and recall 76.00%. Top 100 captures 56 frauds. The difference reflects different review policies.

## Deliverables

- `reports/T085_Phase1_IEEE_Report.pdf` and `reports/ieee_report/main.tex`
- `docs/T085_Phase1.pptx` and `docs/T085_Phase1.pdf`
- `reports/literature_review.md`, `reports/robustness_analysis.md`, `reports/error_analysis.md`
- Seven executed notebooks, the frozen split file and versioned numerical outputs
- `guides/VIVA_GUIDE.md`, `guides/PHASE1_ONLY.md`, `docs/FINAL_AUDIT.md`

## Regenerate the report

```bash
python scripts/update_deliverables.py
python scripts/build_report.py
tectonic reports/ieee_report/main.tex
python scripts/package_report.py
```

The packaging command copies the compiled PDF to the submission path and rebuilds the matching Overleaf archive. After all artifact/source changes, run `python scripts/record_source_manifest.py` and both verification commands above. `scripts/build_presentation.mjs` recreates the editable deck and matching PDF using the bundled runtime described in its header. Presentation charts also remain manually editable. Re-run and inspect all exported artifacts after changing results.

## Limitations

Only two historical days and 75 test frauds limit generalization. Exact duplicates are retained because transaction IDs are absent. AIC/BIC optimize density fit, not fraud recall. This is a bounded staged search, not a global optimum. Robustness studies use fixed K=4 and evaluate validation quality; they are not a comprehensive architecture search. Continuous density is not a fraud posterior. AP intervals condition on observed class counts and omit temporal dependence and refitting uncertainty. Test reuse must be disclosed in Phase 2.

Scope follows the faculty [Phase 1 instructions](https://docs.google.com/document/d/1ZBr-3LoIEFzXz1pxziGqxSVewOAI3sacAzXRaHxtF_Q/edit). Literature citations are in the report and literature review.

### Automated public checks

The [Phase 1 workflow](../../actions/workflows/phase1-checks.yml) runs unit/integration tests and checks the committed report, notebooks, presentation and provenance without the raw dataset. Full numerical reproduction uses the original Kaggle CSV locally.
