# Phase 2 contribution record and future work

This repository already contains the Phase 2 comparison, validation study, hybrid score, two executed notebooks, report and presentation. Do not split or relabel existing commits to create artificial contribution history.

## Actual Phase 2 commits

| Commit | Recorded author | Work |
|---|---|---|
| `728996a` | Aaditya account | Initial undercomplete denoising autoencoder comparison, measured outputs, report and slides |
| `5f8afa8` | Ansh | Expanded four-budget comparison, 12 validation-only ablations, executed triage notebook, report and slides |
| `88bf8bf` | Ansh | Validation-selected hybrid score, executed fusion notebook, tests, report and slides |

The merge commits `f00cb5e` and `79e9468` integrated Ansh's changes into main. These records document repository activity, not an assertion that anyone else wrote another person's work.

## Optional genuine follow-up work

- Gauri: audit the frozen data membership and provide a verified data note.
- Farhana: review preprocessing fit scope and add a leakage check if a concrete gap exists.
- Aaditya: review GMM score alignment and integration against the latest Phase 2 results.
- Bhumi: independently review evaluation and any predeclared validation-only neural improvement.
- Anshvardhan: maintain the hybrid and robustness interpretation, and clarify error cases.

Each person should commit only changes they actually make. A new, genuinely unseen later-time dataset would be needed to close the independent-holdout limitation.
