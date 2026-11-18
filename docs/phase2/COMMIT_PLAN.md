# Phase 2 contribution and commit plan

The integrated Phase 2 experiment was implemented in this checkout. Git authorship must reflect actual work; teammates should review and contribute their own follow-up changes rather than splitting or rewriting this work under their names.

1. Gauri: review `run_phase2.py` data loading and `results/phase2/run_metadata.json`; commit any verified data/split corrections and a short data audit.
2. Farhana: review the use of `src/features.py` and frozen scaler inputs; commit any tested preprocessing corrections and leakage analysis.
3. Aaditya: review the frozen GMM score comparison in `run_phase2.py`; commit any validated baseline/protocol fixes.
4. Bhumi: review the autoencoder architecture, fit and `results/phase2/metrics.csv`; commit tested neural/evaluation improvements with validation-only decisions.
5. Anshvardhan: review `reports/phase2_ieee/main.tex`, the PR figure, error analysis and limitations; commit supported edits and figures.
6. Aaditya as integrator: after actual team changes, rerun Phase 2, rebuild the IEEE PDF and PPTX/PDF, check the results and merge the reviewed branches. Do not attribute this initial integrated commit to teammates who did not make it.

Suggested future commit messages: `docs(data): audit frozen Phase 2 membership`; `test(features): verify legitimate-only transform`; `fix(gmm): validate frozen score alignment`; `feat(autoencoder): improve validation-selected architecture`; `docs(phase2): expand error analysis`; `chore(integration): refresh Phase 2 deliverables`.
