# Team T085: viva preparation

All members should understand the complete protocol as well as their assigned role. Role 1 Gauri: data/splits. Role 2 Farhana: preprocessing. Role 3 Aaditya: GMM. Role 4 Bhumi: metrics. Role 5 Anshvardhan: robustness, error analysis and integration narrative.

## Core explanations

1. **Why novelty detection?** Normal records are plentiful, while fraud patterns can change. The assumption is that useful frauds deviate from legitimate behavior. Unusual legitimate transactions and normal-looking fraud limit this assumption.
2. **Is fitting label-free?** Labels filter the known legitimate training population. No labels enter model inputs or loss. Call it clean-normal novelty detection, not a method with no label access.
3. **What is a GMM?** A weighted sum of Gaussian densities with weights summing to one. The anomaly score is negative log density. Density is not a fraud probability and may exceed one.
4. **How does EM fit it?** The E-step calculates soft component responsibilities. The M-step updates weighted means, covariance matrices and mixing weights. Multiple starts reduce local-optimum risk without guaranteeing a global optimum.
5. **What do covariance types mean?** Full models correlations per component, diagonal assumes component-wise axis-aligned covariance, tied shares a full matrix, and spherical uses a scalar variance per component.
6. **What are AIC and BIC?** AIC=2p-2 log L; BIC=p log n-2 log L. Compare candidates on identical observations in the same feature space. Negative criteria are possible for continuous densities. BIC is the fixed primary selection rule.
7. **What was searched?** Twenty-eight structural candidates cross seven component counts and four covariance types. Predetermined one-factor trials change regularization, starts and iteration caps on the structural winner. This is bounded staged tuning rather than an exhaustive grid.
8. **What was selected?** 20 full-covariance components, reg_covar=1e-06, n_init=2, max_iter=300. Training BIC=-13246003.8. Report the grid-boundary limitation if applicable.
9. **Why regularize covariance?** Adding positive diagonal variance reduces singularity risk. The selected largest condition number is 17,759,113.4. Convergence alone does not establish good conditioning, and good conditioning alone does not establish fraud utility.
10. **How are splits frozen?** Original source row IDs, all three chronological partitions and legitimate training membership are saved in split_indices.npz. Loading checks dataset SHA, fractions, complete/nonoverlapping membership, chronology and normal labels. Equal timestamps stay together.
11. **What is preprocessing?** V1-V28 plus log1p(Amount), fitted legitimate-training StandardScaler, and optional training-fitted PCA for diagnostic experiments. Time defines ordering and EDA. Class never enters X. Save and reuse the full pipeline.
12. **Why not accuracy?** With rare fraud, majority classification can have high accuracy while catching no fraud. Precision, recall, FPR, AP and workload address the review problem. Accuracy is not an evaluation output.
13. **AP versus PR AUC?** AP sums recall increments weighted by precision. Trapezoidal PR AUC joins curve points linearly. GMM AP=0.5822; trapezoidal area=0.5787. Use each split's own prevalence for the no-skill reference.
14. **How do threshold ties work?** At alpha, allow floor(alpha*n_normal_validation) false positives. Use the next descending normal score as cutoff and flag strictly greater scores. Ties make validation calibration conservative. Test FPR can differ.
15. **Explain the 0.5% FPR policy.** Frozen cutoff=63.5486. Test TP=57, FP=175, FN=18, TN=56711. Recall=76.00%, precision=24.57%, actual FPR=0.308% and alerts=232.
16. **Explain capacity.** Top0.5% reviews 284 scored test records and catches 57 frauds at 20.07% precision. Top100 catches 56. Queue membership uses scores with stable chronological ties, never labels. Capacity is distinct from a frozen threshold.
17. **What robustness was completed?** 26 validation-only fits compare four dimensions, four covariance types, three seeds and nested training fractions at fixed K=4. Sample-size studies keep full-training preprocessing fixed. Raw AIC/BIC cannot rank changed dimensions or fitting populations.
18. **What do seed metrics show?** AP mean/sample SD measure metric variation. Spearman correlations measure ranking agreement. Per-row raw score SD measures scale-sensitive score variation. Three seeds and 57 validation frauds limit conclusions.
19. **What does error analysis establish?** It describes missed frauds, high-score legitimate alerts, group amounts/times and standardized feature deviations after the freeze. It cannot identify causes from anonymized features and does not retune the model. At the default error policy there are 3.07 false positives per caught fraud.
20. **Is test evaluation independent?** The initial run already inspected this test partition. The expansion keeps train-only selection and validation-only diagnostics but reports test reuse. It must not be described as a fresh untouched holdout.
21. **What does the bootstrap interval mean?** GMM AP interval=0.4634-0.7123 from500 stratified resamples. It conditions on observed prevalence and omits refitting and temporal dependence. It is not a paired superiority test.
22. **What is left for Phase 2?** Fit an undercomplete/denoising autoencoder on exactly the same normal training membership and preprocessing. Select neural settings on validation, reuse the metrics harness, disclose test reuse and justify any hybrid using ablation evidence.

## Demonstration sequence

Run `python -m unittest discover -s tests -v`, inspect split membership and preprocessing, explain the selected grid row, recompute the default confusion metrics by hand and open notebooks 04-06. Run `python scripts/verify_outputs.py` to check the actual score and metric contracts. Individual members must prepare their own explanations and record real review/contribution evidence.
