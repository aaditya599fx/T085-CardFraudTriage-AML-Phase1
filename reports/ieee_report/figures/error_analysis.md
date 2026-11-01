# Post-freeze GMM error analysis

Policy: 0.5% validation FPR budget, frozen threshold 63.548622; flag strictly larger scores. This descriptive test inspection follows the freeze and never feeds back into fitting or model choice.

| Group | Rows | Median amount | Median score |
|---|---:|---:|---:|
| caught_fraud | 57 | 2.00 | 155.016 |
| false_positive | 175 | 157.43 | 71.365 |
| missed_fraud | 18 | 7.79 | -20.505 |
| true_negative | 56711 | 17.99 | -42.657 |

The model catches 57 frauds and misses 18, with 175 false positives. There are 3.07 legitimate alerts per caught fraud and 4.07 total reviews per caught fraud at this policy.

`missed_frauds.csv` lists every false negative with its original source-row ID, time, amount, label and score. `top_false_positives.csv` lists the twenty highest-scored legitimate alerts. Group summaries report amount quartiles and median elapsed time. `error_feature_deviations.csv` compares absolute standardized feature deviations relative to legitimate-training means and standard deviations.

Largest average absolute standardized deviations by group:

- missed_fraud: V14 (2.76), V17 (2.15), V3 (1.74)
- false_positive: V20 (6.93), V28 (6.56), V23 (6.36)

These are descriptive feature differences, not causal explanations or per-feature GMM likelihood attributions. The anonymized V features have no available business semantics. A high legitimate score can reflect rare normal behavior, and missed fraud can occupy regions that resemble normal training density.

`review_workload.csv` reports all models and configured operating points. A blank false-positives-per-fraud ratio means no fraud was caught; the accompanying boolean makes that explicit. Capacity ranking and fixed score thresholds can produce different alert counts. The review workload is an academic proxy because costs and investigator outcomes are unavailable.
