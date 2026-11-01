# Model-independent evaluation

Bhumi owns evaluation review under the team's role assignment.

`evaluate_scores(y_true, anomaly_score, thresholds=..., review_fractions=..., review_counts=...)` returns ranking metrics, frozen-threshold confusion metrics and capacity metrics. Scores must be finite one-dimensional values with higher scores indicating greater anomalousness. Both classes must occur in the evaluation population.

Average precision is the recall-increment-weighted precision summary. `pr_auc_trapezoidal` integrates the PR curve linearly. Their values differ, so names remain explicit. The PR reference line uses each split's own fraud prevalence. ROC AUC is secondary, and accuracy is not an evaluation output.

For an FPR budget alpha, sort legitimate validation scores descending. With m=floor(alpha*n_normal), use score[m] as the cutoff, and flag score>cutoff. Strict comparison makes ties conservative. Freeze these thresholds before evaluating test labels. Report test FPR rather than assuming the validation budget transfers exactly.

Capacity policies review the top floor(q*n) transactions, or top 100, after sorting by descending score. Stable chronological order breaks ties. Membership depends on scores only. Capacity ranking is a batch policy, distinct from the frozen scalar threshold.

Report TP, FP, FN, TN, recall, precision, FPR, alert count, review fraction, false positives per caught fraud and total reviews per caught fraud. When TP=0, workload ratios are undefined, left blank in CSV, and accompanied by an explicit no_frauds_caught flag.

AP intervals use 500 stratified bootstrap resamples with the observed class counts. They omit model-refitting uncertainty, prevalence shifts and temporal dependence. They are descriptive intervals, not a paired statistical superiority test.

The test partition was inspected in the initial run. The expanded experiment discloses test reuse and makes no fresh-holdout claim. The robustness study uses validation only and cannot alter model choice through test AP.

Frozen threshold CSV files include a hexadecimal float representation. Always use `src.metrics.load_thresholds` to restore the exact cutoff; a default decimal CSV parser may move a tied score boundary.
