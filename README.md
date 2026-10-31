# Project 26: Card Fraud Triage

**Team T085 | Section E | Phase 1: Advanced Machine Learning**

Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection.

## Current status

This is the initial project scaffold. Implementation, experiments, tests and submission artifacts will be added through reviewed team contributions. No GMM implementation or model results are included in the initial commit. Phase 1 focuses on classical machine learning; the autoencoder belongs to Phase 2.

## Planned approach

Fit a Gaussian Mixture Model on legitimate transactions, compare AIC/BIC configurations, and use negative log density as the anomaly score. Evaluate mixed chronological validation/test partitions using precision-recall metrics, false-positive budgets and review capacity. Fit preprocessing on legitimate training only and preserve split membership for later comparisons.

## Team responsibilities

| Role | Responsibility |
|---|---|
| 1 | Data quality, EDA and splits |
| 2 | Preprocessing and feature engineering |
| 3 | GMM modeling |
| 4 | Evaluation and thresholding |
| 5 | Robustness, error analysis and documentation |

The maintainer coordinates initial setup and final integration. The detailed person-by-person assignment is distributed separately. Roles identify ownership and review responsibility; actual contributions will be recorded with commit/PR evidence.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the original [ULB/Kaggle credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/raw/`. The raw CSV and fitted joblib files are excluded from Git. Reproduction commands will be added with the implementation.

## Directory structure

```text
configs/                 Shared setup and team metadata
src/                     Implementation modules
notebooks/               Ordered analysis notebooks
tests/                   Protocol and model checks
data/raw/                Local original CSV, excluded from Git
data/processed/          Saved split membership and provenance
results/figures/         Experiment plots
results/tables/          Evaluation and sensitivity tables
results/scores/          Row-aligned anomaly scores
results/models/          Locally regenerated fitted objects
docs/                    Protocols, presentation and audit
reports/ieee_report/     IEEE report source and figures
scripts/                 Reproduction and artifact generation
guides/                  Scope and individual viva preparation
```

## Contribution workflow

Use your own branch and GitHub account. Review, reproduce and adapt the files you own, commit the actual changes, and request another member's review. Coordinate shared configuration and result files to avoid conflicting versions. Do not commit a completed-results claim before the corresponding work has been integrated and checked.
