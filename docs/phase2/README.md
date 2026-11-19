# Phase 2 fraud triage

The [executed triage notebook](../../notebooks/phase2/01_fraud_triage.ipynb), [IEEE report](../../reports/phase2_ieee/main.pdf), and [slides](T085_Phase2.pptx) cover the comparison and the ablations. The notebook runs from the committed tables and curves. The original ULB/Kaggle CSV is needed to regenerate them.

## Reproduction

```sh
python -m pip install -r requirements.txt
python run_phase2_study.py --data /path/to/creditcard.csv
python run_phase2_hybrid.py --data /path/to/creditcard.csv
python -m unittest discover -s phase2/tests -v
```

The study verifies the original CSV and split membership. It reads the frozen Phase 1 GMM and original Phase 2 autoencoder scores and writes to `results/phase2/study/`. Its metadata records package versions and input hashes. `python run_phase2.py` remains the original baseline training command and writes the original Phase 2 output paths. Use a separate checkout if rerunning the original baseline so the frozen comparison files stay available.

To rebuild the report and slide deck from the repository root, use Tectonic and local Node.js dependencies.

```sh
(cd reports/phase2_ieee && tectonic main.tex)
npm install --no-save pptxgenjs jszip
node scripts/build_phase2_slides.mjs
```

## Evaluation protocol

- Compute AP, trapezoidal PR AUC, ROC AUC and the fraud base rate. Use a threshold based on legitimate validation scores for each model and each FPR budget of 0.1%, 0.5%, 1% and 5%. Report realized FPR, recall, precision and false alerts. At a fixed review capacity, compare the models on equally sized queues.
- Run one-factor ablations with bottleneck widths 4, 8 and 16 and with noise standard deviation 0 or 0.05 at width 8. Pair seeds 13, 26 and 42 across variants. Fit the scaler on the first approximately 90% of legitimate training rows in chronological order. Reserve the later legitimate training rows for checkpoint selection and keep equal times together.
- Use ReLU hidden layers, a linear output, Adam, batch size 512, learning rate 0.001 and L2 coefficient 0.0001. Add fresh Gaussian noise per epoch in denoising trials. Choose the lowest clean reconstruction MSE on the reserve with patience 5 and a cap of 35 epochs. Score each checkpoint on the same mixed validation partition and summarize AP across seeds. The ablation code receives training and validation only.
- The frozen historical autoencoder used fixed corruption generated before fitting and scikit-learn's internal validation R² with a random reserve from legitimate training. The new controlled ablation protocol uses a shared chronological reserve and clean MSE. Treat its scores as validation evidence for the tested variants.

## Hybrid score fusion

The [executed hybrid notebook](../../notebooks/phase2/02_hybrid_fusion.ipynb) combines low GMM density and high autoencoder reconstruction error. For each model, convert its raw score to the negative log of the empirical upper-tail proportion among legitimate validation transactions, with one added to the numerator and denominator. This puts both scores on a common scale and limits scores above the largest legitimate reference value to the finest tail resolution supported by the reference sample.

Choose the autoencoder weight from 0.05, 0.10, 0.20, 0.35 and 0.50 by mixed-validation AP. Weight the GMM by the remaining share. Use the lower autoencoder weight if AP ties. Compare the resulting hybrid against both frozen standalone scores to measure the effect of removing either component. Each score gets its own cutoffs from the same legitimate validation sample and is evaluated at the same FPR budgets and review capacities.

The 95% GMM and 5% autoencoder hybrid reaches 0.6327 validation AP versus 0.6299 for GMM. On the reused historical test, it reaches 0.5754 AP versus 0.5822 for GMM. Both catch 57 of 75 frauds in 284 reviews. At the 1% validation FPR budget, both catch 58 frauds and the hybrid produces 320 false alerts versus 324 for GMM. The descriptive paired AP-difference interval spans zero. These results support a small validation gain and similar historical triage performance. Use `results/phase2/hybrid/` for the full operating-point tables and input hashes.

## Holdout status

The original 56,961-row test partition was inspected in Phase 1. All metrics labeled historical test refer to that same partition. Topic 26's untouched-test condition remains open. To close it, document a genuinely unseen later batch with compatible feature definitions and the same PCA representation. Freeze the models and validation-derived policies before examining the batch labels. Evaluate both methods on that batch with the same score orientation and review budgets.

Each member can add their own verified commit, PR or review and demo notes to the individual review record in [CONTRIBUTIONS.md](../../CONTRIBUTIONS.md).
