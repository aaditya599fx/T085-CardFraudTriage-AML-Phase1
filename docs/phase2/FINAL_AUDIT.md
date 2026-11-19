# Phase 2 submission audit

Checked against main after Ansh's hybrid merge (`79e9468`). The repository includes the original autoencoder comparison, a controlled 12-run validation-only ablation study, a validation-selected hybrid, two executed Phase 2 notebooks, an IEEE-format report, a 16-slide presentation and the corresponding results tables and figures. The public Phase 2 GitHub Actions workflow passes.

The initial autoencoder test AP is 0.0908, the frozen GMM test AP is 0.5822, and the hybrid test AP is 0.5754. The hybrid's validation AP is 0.6327, slightly above GMM's 0.6299, but its historical test AP is slightly below GMM. At 284 test reviews, hybrid and GMM each catch 57 of 75 frauds. The hybrid changes the project from a two-model comparison to a three-score comparison; it does not change the conclusion that the frozen GMM remains the strongest standalone model on this historical test.

The original test partition was inspected in Phase 1. These Phase 2 findings are follow-up analyses rather than a clean independent holdout. No new later-time labelled batch is included. The hybrid weight was selected on mixed validation labels, which is acceptable for the stated validation policy but should be described accurately as supervised fusion selection. The underlying GMM and autoencoder were fitted on legitimate training data.

The committed Phase 2 workflows test artifact consistency and execute the notebooks without the original CSV. Independent reproduction of model training still requires the original ULB/Kaggle CSV. The local audit used an environment with `nbformat`; the system Python alone lacked that notebook dependency.
