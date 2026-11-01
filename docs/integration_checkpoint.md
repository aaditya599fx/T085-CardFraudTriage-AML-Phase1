# Integration checkpoint

This commit supplies the integration owner's orchestration, output verifier and notebook execution driver. It does not add or replace the other roles' modules.

## Repository state at this checkpoint

Data/splits, preprocessing, GMM and classical-comparator modules are present. The official branch does not yet contain these required interfaces:

- `src/metrics.py` and `src/evaluation.py` (Role 4).
- `src/error_analysis.py` and `src/robustness.py` (Role 5).

The existing shared configuration already contains their planned settings, so it is preserved. Full benchmark reproduction is pending those module merges. Final documentation, presentation and submission auditing remain later stages of the contribution plan.

## Commands

```bash
# Read-only check; exits with a list of missing prerequisites until they are merged.
python run_all.py --check-only

# After all modules and the original dataset are available:
python run_all.py
python -m unittest discover -s tests -v
python scripts/verify_outputs.py
python scripts/execute_notebooks.py
```

An external CSV can be supplied to both runner and verifier with `--data /absolute/path/creditcard.csv`. Both also support `--config`. A quick run uses `--skip-robustness`, and its metadata explicitly says `quick_run_complete`. Verify that mode with the same flag; it does not certify a complete Phase 1 reproduction.

## Integration behavior

- Help and preflight remain usable before the pending modules exist.
- Frozen chronological row IDs are reused and checked against the dataset and configured split policy.
- Preprocessing and every primary model receive the same legitimate training population.
- Model selection uses training information criteria. Validation freezes the thresholds; test reuse is disclosed.
- Score arrays include source-row IDs and must be finite one-dimensional arrays of the expected length.
- The model-independent evaluation, error-analysis and validation-only robustness interfaces are called in order.
- `results/experiment_plan.json` is generated from the actual run's config, dataset and split hashes.
- `results/run_metadata.json` is written on successful completion, with a distinct status for a quick run. A failed in-progress run must not retain a stale success marker for that run.
- Verification recomputes ranking, confusion and capacity metrics, checks configuration/provenance, and checks error-group and robustness coverage.
- The notebook driver validates and executes each available notebook using the current Python interpreter.

No private-run metadata is copied into this official checkpoint. The official plan/run metadata will be created by an actual pipeline run after the missing modules are integrated.

## Checks performed

- Six integration-owner unittest cases cover CLI help, config validation, read-only preflight, missing-role diagnostics and quick-run dependency behavior.
- A separate isolated synthetic fixture exercised the entire runner using already available interface implementations. Its complete pipeline and output-verification checks passed. Deliberately reversed score-row IDs were rejected.
- The notebook driver was tested against the four currently available notebooks in that isolated fixture.

The fixture uses 1,200 synthetic rows and a reduced experimental configuration. It validates integration control flow and interfaces; it is not a rerun of the original ULB benchmark and produces no official benchmark claim. Fixture files and pending-role implementations are not part of this commit.
