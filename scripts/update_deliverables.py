"""Synchronize README, analytical narratives and IEEE source with the recorded full experiment.

Run after run_all.py, then build_report.py and compile main.tex. This script does not train,
select or evaluate a model. It only consumes saved tables and fills report text.
"""
from pathlib import Path
import json
import math
import re
import pandas as pd

R=Path(__file__).resolve().parents[1]
T=R/'results/tables'
def put(path,text):
    (R/path).write_text(text.strip()+'\n')
def section(s,start,end,replacement):
    a=s.index(start);b=s.index(end,a)
    return s[:a]+replacement+'\n\n'+s[b:]
def pct(x,digits=2):return f'{100*x:.{digits}f}'

def main():
    grid=pd.read_csv(T/'gmm_grid.csv');selected=grid[grid.selected].iloc[0]
    rank=pd.read_csv(T/'ranking_metrics.csv');test=rank[rank.split=='test'].set_index('model')
    op=pd.read_csv(T/'operating_points.csv');gop=op[(op.model=='gmm')&(op.split=='test')]
    point=gop[gop.fpr_budget==.005].iloc[0]
    capacity=pd.read_csv(T/'review_capacity.csv')
    cap=capacity[(capacity.model=='gmm')&(capacity.split=='test')&(capacity.policy=='fraction')&(capacity.capacity_fraction==.005)].iloc[0]
    count=capacity[(capacity.model=='gmm')&(capacity.split=='test')&(capacity.policy=='count')].iloc[0]
    seed=pd.read_csv(T/'seed_summary.csv');rob=pd.read_csv(T/'robustness_trials.csv')
    corr=pd.read_csv(T/'seed_rank_correlations.csv');err=pd.read_csv(T/'error_group_summary.csv')
    cond=float(selected.max_cov_condition)
    reg_tex='10^{'+str(int(round(math.log10(selected.reg_covar))))+'}'
    best_aic=grid[grid.converged & grid.aic.notna()].sort_values('aic').iloc[0]
    same=bool(best_aic.name==selected.name)
    diag=rob[rob.study=='dimensions_covariance'];full=diag[diag.covariance=='full']
    size=rob[(rob.study=='training_size')|((rob.study=='dimensions_covariance')&(rob.dimensions==29)&(rob.covariance=='full'))].sort_values('train_fraction')
    failed=int((~rob.converged).sum());gridfailed=int((~grid.converged).sum())
    result=dict(components=int(selected.components),covariance=selected.covariance,
        reg_covar=float(selected.reg_covar),n_init=int(selected.n_init),max_iter=int(selected.max_iter),
        test_ap=float(test.loc['gmm','average_precision']),test_pr_auc=float(test.loc['gmm','pr_auc_trapezoidal']),
        test_ap_ci=[float(test.loc['gmm','ap_ci_lower']),float(test.loc['gmm','ap_ci_upper'])],
        selected_condition=cond,primary_candidates=len(grid),robustness_trials=len(rob),
        tests=14,notebooks=7,test_reused=True,split_rows=56961)
    put('results/submission_summary.json',json.dumps(result,indent=2))
    roles='''| Role | Member | Responsibility |
|---|---|---|
| 1 | Gauri mishra | Data, training EDA and saved chronological splits |
| 2 | Farhana Pervin | Preprocessing, feature decisions and PCA |
| 3 | Aaditya Yadav | GMM selection and integration |
| 4 | Bhumi Rastogi | Metrics, threshold calibration and review capacity |
| 5 | Anshvardhan Badkur | Robustness, error analysis and final narrative |'''
    put('README.md',f'''# Card Fraud Triage: GMM Density Scoring vs Autoencoder Anomaly Detector

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
jupyter lab
```

Dataset: [ULB/Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Original CSV: 284,807 rows, 492 frauds, no missing values. The CSV and fitted joblib models are excluded from Git. `results/models/README.md` documents regeneration. An external CSV path works with `python run_all.py --data /absolute/path/creditcard.csv`.

The default command executes the full project. `--skip-robustness` is only a quick rerun. Saved split indices are reused and checked against the dataset checksum, configured fractions, chronology and legitimate training membership. Mismatches fail instead of silently changing the experiment.

## Team responsibilities

{roles}

Assignments are ownership and review responsibilities. Actual personal contributions require real commit/review evidence in `CONTRIBUTIONS.md`.

## Repository structure

```text
configs/                 Fixed experiment configuration and team metadata
src/                     Data, preprocessing, GMM, baselines, metrics, robustness, error analysis
tests/                   Fourteen protocol and behavior checks
notebooks/               Seven executed notebooks, including an assembled Phase 1 summary
data/processed/          Frozen split indices, dataset checksum and split manifest
results/figures/         EDA, model selection, validation/test PR, sensitivity and error plots
results/tables/          Selection, triage, robustness, workload and error-analysis results
results/scores/          Row-aligned validation/test score arrays
results/models/          Model regeneration guide; fitted binaries remain local
docs/                    Protocols, data card, audit, editable presentation and PDF
reports/                 Literature, robustness, error analysis and compiled IEEE report
reports/ieee_report/     Compilable IEEEtran source and referenced figures
scripts/                 Narrative/table synchronization and report rebuilding
guides/                  Phase 1 checklist and viva preparation
run_all.py               Complete reproducible experiment
```

## Experimental protocol

- Approximate chronological 60/20/20 partitions, preserving equal timestamps. There are 170,528 legitimate training records, 57 validation frauds and 75 test frauds.
- Explicit `data/processed/split_indices.npz` fixes membership for every model and Phase 2. Provider-created PCA features cannot be independently refit.
- Inputs are V1-V28 and log1p(Amount). Time defines the split and EDA. Preprocessing fits on legitimate training only.
- Structural search crosses K={{1,2,4,8,12,16,20}} and four covariance types. Predetermined one-factor trials vary regularization, initialization count and iteration limit on the structural winner. Minimum converged training BIC chooses the final GMM. AIC is reported separately.
- The same training rows feed PCA reconstruction, Isolation Forest and an explicitly approximate RBF one-class SVM. Baseline settings stay fixed.
- Validation calibrates strict score>threshold policies at FPR budgets 0.1%, 0.5%, 1% and 5%. Capacity policies review the top 0.1%, 0.5%, 1% or top 100 scores, with stable chronological tie breaking.
- Robustness uses validation only: dimensions 5/10/20/29, all four covariance types, three seeds and nested 10%/50%/100% training subsets at fixed K=4. Raw AIC/BIC are not compared across dimensional spaces or different fitting populations.
- Post-freeze error analysis describes missed frauds, top false positives and review workload. It does not retune the model.

**Test reuse disclosure:** this is an expanded follow-up to the original Phase 1 run. The same test partition was already inspected. Model selection still uses training BIC, robustness uses validation, and thresholds freeze before the current test computation, but these results are not a fresh independent holdout. The initial run is preserved in Git history at commit `753087b`.

## Recorded results

Training BIC selected **{int(selected.components)} {selected.covariance}-covariance components**, reg_covar={selected.reg_covar:g}, n_init={int(selected.n_init)} and max_iter={int(selected.max_iter)}. {len(grid)} primary fits and {len(rob)} robustness fits are recorded. The selected covariance condition number is approximately {cond:,.0f}.

| Model | Test AP | Trapezoidal PR AUC |
|---|---:|---:|
'''+ '\n'.join(f'| {name} | {row.average_precision:.4f} | {row.pr_auc_trapezoidal:.4f} |' for name,row in test.iterrows())+f'''

GMM AP's descriptive 95% stratified bootstrap interval is {test.loc['gmm','ap_ci_lower']:.4f}-{test.loc['gmm','ap_ci_upper']:.4f}. At the **0.5% validation FPR budget**, GMM catches **{int(point.tp)}/75 frauds**, with **{int(point.fp)} false positives**, **{pct(point.fpr,3)}% actual test FPR**, and **{pct(point.precision)}% precision** across **{int(point.flagged)} alerts**.

The separate top-0.5% capacity policy reviews {int(cap.k)} records and catches {int(cap.tp)} frauds, precision {pct(cap.precision_at_k)}% and recall {pct(cap.recall_at_k)}%. Top 100 captures {int(count.tp)} frauds. The difference reflects different review policies.

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
```

Copy the compiled `main.pdf` to `reports/T085_Phase1_IEEE_Report.pdf` for submission. `scripts/build_presentation.mjs` recreates the editable deck with the bundled Artifact Tool runtime described in its header. Presentation charts also remain manually editable. Re-run and inspect all exported artifacts after changing results.

## Limitations

Only two historical days and 75 test frauds limit generalization. Exact duplicates are retained because transaction IDs are absent. AIC/BIC optimize density fit, not fraud recall. This is a bounded staged search, not a global optimum. Robustness studies use fixed K=4 and evaluate validation quality; they are not a comprehensive architecture search. Continuous density is not a fraud posterior. AP intervals condition on observed class counts and omit temporal dependence and refitting uncertainty. Test reuse must be disclosed in Phase 2.

Scope follows the faculty [Phase 1 instructions](https://docs.google.com/document/d/1ZBr-3LoIEFzXz1pxziGqxSVewOAI3sacAzXRaHxtF_Q/edit). Literature citations are in the report and literature review.
''')
    seedlines='\n'.join(f'| {r.covariance} | {int(r.runs)} | {r.validation_ap_mean:.4f} | {r.validation_ap_std:.4f} | {r.maximum_condition:,.1f} |' for _,r in seed.iterrows())
    sizelines='\n'.join(f'| {r.train_fraction:.0%} | {int(r.fit_rows):,} | {r.validation_average_precision:.4f} | {r.max_cov_condition:,.1f} |' for _,r in size.iterrows())
    put('reports/robustness_analysis.md',f'''# Robustness and stability analysis

{len(rob)} predetermined trials use legitimate training data and mixed validation only. Test observations never enter this module. K=4, n_init=2, max_iter=200 and reg_covar=1e-5 are fixed for the diagnostic studies.

## Dimensionality and covariance

Cross dimensions 5,10,20,29 with diagonal, full, tied and spherical covariance. PCA and StandardScaler fit on legitimate training only. Full covariance validation AP across dimensions is '''+', '.join(f'{int(r.dimensions)}d: {r.validation_average_precision:.4f}' for _,r in full.iterrows())+f'''.

Maximum covariance condition numbers, minimum covariance eigenvalues, component weights, warnings and convergence are recorded per trial. {failed} of {len(rob)} robustness trials did not converge or failed numerically. A covariance condition number above 1e6 is flagged diagnostically; the cutoff is not a proof of invalidity. Spherical covariance has condition number one within each component by construction, which does not establish good ranking or comparable variance scales between components.

Changing dimension affects both retained information and conditioning. A drop in condition number cannot by itself prove a better fraud detector. Raw AIC/BIC must not rank these different feature spaces.

## Multiple seeds

All four covariance types at full dimension are repeated with seeds 13,26,42.

| Covariance | Runs | Mean validation AP | Sample SD AP | Maximum condition |
|---|---:|---:|---:|---:|
{seedlines}

Pairwise validation score-rank correlations range from {corr.validation_spearman.min():.4f} to {corr.validation_spearman.max():.4f}. `seed_score_dispersion.csv.gz` stores per-observation mean and sample SD of raw negative log density across seeds. Raw-score SD depends on score scale; rank correlations help interpret ordering stability.

## Training size

Nested seeded 10%,50%,100% subsets isolate GMM fitting-population size at 29 dimensions with full covariance. The full legitimate-training preprocessing is held fixed, so this is a GMM sample-size experiment, not a full-pipeline learning curve.

| Fraction | Fit rows | Validation AP | Maximum condition |
|---|---:|---:|---:|
{sizelines}

## Interpretation limits

These are descriptive sensitivity experiments, not a statistical guarantee of stability. Some covariance assumptions constrain condition numbers by construction. Validation has only 57 frauds, seeds are only three, and K=4 is fixed here. The primary model uses a separate train-only structural/optimizer selection. Do not choose a winner across dimensions from the raw density criteria or use test outcomes to tune these experiments.
''')
    errorlines='\n'.join(f'| {r.group} | {int(r.rows)} | {r.median_amount:.2f} | {r.median_score:.3f} |' for _,r in err.iterrows())
    dev=pd.read_csv(T/'error_feature_deviations.csv')
    group_findings=[]
    for group in ['missed_fraud','false_positive']:
        rows=dev[dev.group==group].sort_values('mean_abs_training_z',ascending=False).head(3)
        group_findings.append(group+': '+', '.join(f'{r.feature} ({r.mean_abs_training_z:.2f})' for _,r in rows.iterrows()))
    put('reports/error_analysis.md',f'''# Post-freeze GMM error analysis

Policy: 0.5% validation FPR budget, frozen threshold {point.threshold:.6f}; flag strictly larger scores. This descriptive test inspection follows the freeze and never feeds back into fitting or model choice.

| Group | Rows | Median amount | Median score |
|---|---:|---:|---:|
{errorlines}

The model catches {int(point.tp)} frauds and misses {int(point.fn)}, with {int(point.fp)} false positives. There are {point.fp/point.tp:.2f} legitimate alerts per caught fraud and {point.flagged/point.tp:.2f} total reviews per caught fraud at this policy.

`missed_frauds.csv` lists every false negative with its original source-row ID, time, amount, label and score. `top_false_positives.csv` lists the twenty highest-scored legitimate alerts. Group summaries report amount quartiles and median elapsed time. `error_feature_deviations.csv` compares absolute standardized feature deviations relative to legitimate-training means and standard deviations.

Largest average absolute standardized deviations by group:

- {group_findings[0]}
- {group_findings[1]}

These are descriptive feature differences, not causal explanations or per-feature GMM likelihood attributions. The anonymized V features have no available business semantics. A high legitimate score can reflect rare normal behavior, and missed fraud can occupy regions that resemble normal training density.

`review_workload.csv` reports all models and configured operating points. A blank false-positives-per-fraud ratio means no fraud was caught; the accompanying boolean makes that explicit. Capacity ranking and fixed score thresholds can produce different alert counts. The review workload is an academic proxy because costs and investigator outcomes are unavailable.
''')
    tex=R/'reports/ieee_report/main.tex';s=tex.read_text()
    abstract=f'''\\begin{{abstract}}
We study card fraud as novelty detection using the original ULB/Kaggle dataset. All models learn from 170,528 legitimate training records under frozen chronological partitions. An expanded train-only search compares seven component counts and four covariance types, followed by predetermined optimizer trials. The selected {selected.covariance}-covariance GMM has {int(selected.components)} components and achieves test average precision {test.loc['gmm','average_precision']:.4f}. At the 0.5\\% validation false-positive-rate budget it catches {int(point.tp)} of 75 test frauds with {int(point.fp)} false positives. We add {len(rob)} validation-only sensitivity fits across dimension, covariance, random seed and training size, explicit saved split membership, and post-freeze error analysis. The same test set was inspected in an earlier run, so these expanded results are follow-up findings rather than a fresh independent holdout. Autoencoder and hybrid training remain Phase 2.
\\end{{abstract}}'''
    s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:abstract,s,flags=re.S)
    s=s.replace('Our contributions are an executable normal-training protocol, a bounded GMM selection study, explicit operational policies and an original evaluation with retained negative results.', 'The expanded study adds persisted split membership, a broader train-only GMM search, validation-only robustness studies and descriptive error analysis. The original test partition has already been inspected; the expanded results explicitly disclose that reuse.')
    model=f'''\\subsection{{Model selection and staged optimization}}
The structural search crosses $K\\in\\{{1,2,4,8,12,16,20\\}}$ with diagonal, full, tied and spherical covariance. All 28 candidates fit identical legitimate training observations in the same 29-dimensional feature space. For $p$ free parameters and training likelihood $L$,
\\begin{{equation}}
\\mathrm{{AIC}}=2p-2\\log L,\\qquad\\mathrm{{BIC}}=p\\log n-2\\log L.
\\end{{equation}}
Full covariance has $(K-1)+Kd+Kd(d+1)/2$ parameters, diagonal $(K-1)+2Kd$, tied $(K-1)+Kd+d(d+1)/2$, and spherical $(K-1)+Kd+K$. Minimum finite BIC among converged fits selects the structural anchor. Predetermined one-factor trials at that anchor vary regularization across $10^{{-6}},10^{{-5}},10^{{-4}}$, initialization count across 1,2,3 and iteration cap across 100,300. Duplicate baseline settings are not refitted. This is a staged search rather than a full Cartesian optimization.

Across {len(grid)} primary fits, {gridfailed} failed or did not converge. The selected model has {int(selected.components)} {selected.covariance}-covariance components, $\\lambda={reg_tex}$, {int(selected.n_init)} initialization(s) and an iteration limit of {int(selected.max_iter)}. Its BIC is {selected.bic:.1f} and AIC {selected.aic:.1f}. AIC and BIC {'select the same recorded fit' if same else 'select different recorded fits; BIC remains the predetermined decision rule'}. A minimum at the largest tested K is a bounded-search result, not a global optimum claim.

The selected model's maximum covariance condition number is {cond:.1f} and minimum component weight is {selected.min_weight:.4f}. EM convergence does not by itself establish stability. Negative information criteria are possible for continuous density. Neither validation fraud labels nor test metrics select the primary GMM. Information criteria are not compared across the dimensionality and sample-size studies because the feature spaces or fitting populations differ.
\\begin{{figure}}[t]
\\centering\\includegraphics[width=\\columnwidth]{{figures/gmm_selection.png}}
\\caption{{Training AIC and BIC for structural candidates. The star marks the selected optimizer fit; all trial diagnostics appear in the selection table.}}
\\end{{figure}}'''
    s=section(s,'\\subsection{Model selection','\\section{Classical Comparators}',model)
    s=s.replace('The test set is reserved for the final metric computation after model and threshold policies have frozen.', 'The current test metric computation follows the model and threshold freeze. However, this test partition was already inspected in the initial Phase 1 run; it is reused here and is not a new independent holdout.')
    s=s.replace('We set $\\lambda=10^{-5}$, allow 300 iterations and use two initializations.', 'The structural grid initially sets $\\lambda=10^{-5}$, allows 300 iterations and uses two initializations; staged trials then vary these settings.')
    s=re.sub(r'The GMM AP interval is .*?These intervals condition',lambda m:f"The GMM AP interval is {test.loc['gmm','ap_ci_lower']:.4f}--{test.loc['gmm','ap_ci_upper']:.4f} and PCA's is {test.loc['pca','ap_ci_lower']:.4f}--{test.loc['pca','ap_ci_upper']:.4f}. These intervals condition",s,flags=re.S)
    s=s.replace('For the test size, $q=0.005$ gives 284 reviews.', 'For the test size, $q=0.005$ gives 284 reviews. We additionally report top-100 capacity. FPR budgets are 0.1\\%, 0.5\\%, 1\\% and 5\\%.')
    s=s.replace('Untouched mixed-test precision-recall curves. The dashed reference shows test fraud prevalence.', 'Validation and reused mixed-test precision-recall curves. Dashed references show each population\'s fraud prevalence.')
    evaluation=f'''\\section{{Evaluation Protocol}}
\\subsection{{Ranking summaries}}
Report AP, trapezoidal PR AUC and secondary ROC AUC. AP weights precision by recall increments; trapezoidal PR area linearly connects curve points \\cite{{ap}}. The no-skill precision reference uses each partition's own fraud prevalence. Test prevalence is 0.001317.

The GMM AP interval is {test.loc['gmm','ap_ci_lower']:.4f}--{test.loc['gmm','ap_ci_upper']:.4f}, from 500 stratified bootstrap resamples retaining class counts. These descriptive intervals omit temporal dependence, prevalence changes and model-refitting uncertainty. They are not a paired superiority test.

\\subsection{{Frozen false-positive policies}}
For budget $\\alpha$, sort legitimate validation scores descending and let $m=\\lfloor\\alpha n_0\\rfloor$. Use the zero-indexed score at rank $m$ as threshold and flag strictly larger scores. This permits at most $m$ validation false positives, conservatively handling ties. Budgets are 0.1\\%,0.5\\%,1\\%,5\\%. Freeze cutoffs before test labels are evaluated, then report realized test FPR rather than assuming exact transfer. Store hexadecimal float cutoffs and reload them exactly, avoiding decimal-parser rounding at tied boundaries.

\\subsection{{Review capacity}}
Separately review the top $k=\\max(1,\\lfloor qn\\rfloor)$ scores for $q\\in\\{{0.001,0.005,0.01\\}}$, plus top 100. Stable chronological order breaks score ties. Labels measure quality after scores select the queue. The top-0.5\\% test capacity is 284 reviews. This batch policy differs from a frozen scalar threshold.
'''
    s=section(s,'\\section{Evaluation Protocol}','\\section{Results and Operational Trade-offs}',evaluation)

    results=f'''\\section{{Results and Operational Trade-offs}}
Table~\\ref{{tab:ranking}} reports the fixed comparator configurations and the train-BIC-selected GMM. GMM test AP is {test.loc['gmm','average_precision']:.4f}, compared with PCA {test.loc['pca','average_precision']:.4f}, Isolation Forest {test.loc['isolation_forest','average_precision']:.4f}, and approximate OC-SVM {test.loc['approx_ocsvm','average_precision']:.4f}. These results describe these configurations and this reused historical split, not exhaustive superiority of a method family.

GMM ROC AUC is {test.loc['gmm','roc_auc']:.4f}, while precision-recall summaries remain sensitive to the rare-event review burden. Approximate OC-SVM's weak result is retained; it does not establish that exact kernel OC-SVM cannot work. Bootstrap intervals do not substitute for a paired significance test.

At the 0.5\\% validation FPR policy, GMM's threshold is {point.threshold:.4f}. Test outcomes are TP={int(point.tp)}, FP={int(point.fp)}, FN={int(point.fn)}, TN={int(point.tn)}. Recall is {pct(point.recall)}\\%, precision {pct(point.precision)}\\%, realized FPR {pct(point.fpr,3)}\\%, and total alerts {int(point.flagged)}. The workload is {point.fp/point.tp:.2f} false positives and {point.flagged/point.tp:.2f} reviews per caught fraud. Other configured policies appear in Table~\\ref{{tab:policy}}.

The separate top-0.5\\% policy reviews {int(cap.k)} records and catches {int(cap.tp)} frauds, with precision {pct(cap.precision_at_k)}\\% and recall {pct(cap.recall_at_k)}\\%. Top 100 captures {int(count.tp)} frauds. Labels measure queue quality after membership is selected from scores. Frozen thresholds and batch capacity policies can produce different workloads.

\\begin{{table}}[t]
\\caption{{Test ranking metrics. AP is the primary summary.}}
\\centering\\small
\\begin{{tabular}}{{lrrr}}\\toprule
Model & AP & PR AUC & ROC AUC \\\\\\midrule
% BEGIN GENERATED ranking
% END GENERATED ranking
\\bottomrule\\end{{tabular}}\\label{{tab:ranking}}
\\end{{table}}
\\begin{{table}}[t]
\\caption{{GMM reused-test outcomes at frozen validation FPR budgets. All rates are percentages.}}
\\centering\\footnotesize
\\begin{{tabular}}{{rrrrrr}}\\toprule
Budget & TP & FP & Recall & Actual FPR & Precision \\\\\\midrule
% BEGIN GENERATED operating
% END GENERATED operating
\\bottomrule\\end{{tabular}}\\label{{tab:policy}}
\\end{{table}}
\\begin{{figure}}[t]
\\centering\\includegraphics[width=\\columnwidth]{{figures/pr_curves.png}}
\\caption{{Validation and reused-test PR curves with population-specific prevalence references.}}
\\end{{figure}}
\\begin{{figure}}[t]
\\centering\\includegraphics[width=\\columnwidth]{{figures/threshold_tradeoffs.png}}
\\caption{{Recall and realized test FPR at frozen validation budgets.}}
\\end{{figure}}'''
    # Replace evaluation section's old floating ranking table too; results now owns all result tables.
    pre=s.index('\\section{Results and Operational Trade-offs}')
    head=s[:pre]
    head=re.sub(r'\\begin\{table\}\[t\]\n\\caption\{Test ranking metrics.*?\\end\{table\}', '',head,flags=re.S)
    s=head+s[pre:]
    s=section(s,'\\section{Results and Operational Trade-offs}','\\section{Reproducibility and Validation}',results)
    reproducibility='''\\section{Reproducibility and Validation}
The source separates data, preprocessing, GMM, baselines, metrics, robustness and error analysis. Seven executed notebooks include an assembled Phase 1 summary. The default \\texttt{python run\\_all.py} executes the full study. A quick skip-robustness mode is explicitly incomplete for reproduction.

The dataset checksum, chronological indices, legitimate training membership and partition index checksums are persisted. Loading refuses a different dataset, changed fractions, invalid membership or altered chronology. The complete preprocessing pipeline, scaler and study PCA objects are saved locally. Model files are excluded from Git; their regeneration guide remains visible in results/models.

Fourteen automated checks cover chronology, ties, normal-only scaling, pipeline reuse, split provenance/tampering, all covariance forms, logged fit failure, staged selection and metric identities. Package versions, configuration hashes and the test-reuse disclosure are saved. Team responsibilities follow the supplied five-role assignment; actual personal contributions need independent review evidence rather than inferred attribution.
'''
    s=section(s,'\\section{Reproducibility and Validation}','\\section{Limitations and Phase 2 Plan}',reproducibility)
    robust=f'''\\section{{Robustness and Error Analysis}}
The validation-only module performs {len(rob)} fits. It crosses dimensions 5,10,20,29 with four covariance types at K=4, then repeats full-dimensional covariance settings across seeds 13,26,42 and evaluates nested training subsets. Preprocessing fits only legitimate training rows. No test frame is accepted by this module.

Full-covariance validation AP across dimensions is '''+', '.join(f'{int(r.dimensions)}d: {r.validation_average_precision:.4f}' for _,r in full.iterrows())+f'''. {failed} diagnostic fits failed or did not converge. All warning and covariance-eigenvalue diagnostics are retained. Condition number above $10^6$ is an explicit diagnostic flag, not a universal validity threshold.

Across the three seeds, covariance-specific AP means and sample standard deviations are '''+', '.join(f'{r.covariance}: {r.validation_ap_mean:.4f} $\\pm$ {r.validation_ap_std:.4f}' for _,r in seed.iterrows())+f'''. Pairwise score-rank correlations range from {corr.validation_spearman.min():.4f} to {corr.validation_spearman.max():.4f}. Raw-score dispersion is also saved per validation observation. Spherical covariance has condition number one by construction, so conditioning alone cannot establish detector quality.

At training fractions 10\\%,50\\%,100\\%, full-covariance K=4 validation AP is '''+', '.join(f'{r.validation_average_precision:.4f}' for _,r in size.iterrows())+'''. Preprocessing stays fit on the full legitimate training set to isolate GMM sample size. Raw information criteria do not select across these changing populations or dimensions.

Post-freeze test inspection saves every missed fraud, the twenty highest-scoring legitimate alerts, group amount/time/score summaries and standardized feature deviations. These quantities are descriptive, not causal feature explanations. Anonymized features prevent business-semantic interpretation. Error analysis never feeds back into model fitting or threshold choice.
\\begin{figure}[t]
\\centering\\includegraphics[width=\\columnwidth]{figures/dimensionality_stability.png}
\\caption{Validation AP and covariance conditioning across dimensionality and covariance assumptions at fixed K=4.}
\\end{figure}
\\begin{figure}[t]
\\centering\\includegraphics[width=\\columnwidth]{figures/error_analysis.png}
\\caption{Post-freeze error groups under the 0.5\\% validation FPR policy. The score panel uses a monotone signed-log display transform.}
\\end{figure}
'''
    s=s.replace('\\section{Limitations and Phase 2 Plan}',robust+'\n\\section{Limitations and Phase 2 Plan}')
    # Idempotence: an existing generated robustness section is replaced on later invocations.
    first=s.find('\\section{Robustness and Error Analysis}')
    second=s.find('\\section{Robustness and Error Analysis}',first+1)
    if second!=-1:s=s[:first]+s[second:]
    a=s.index('\\section{Limitations and Phase 2 Plan}');b=s.index('\\begin{thebibliography}',a)
    s=s[:a]+'''\\section{Limitations and Phase 2 Plan}
Only two historical days and 75 test frauds limit generalization. The test set was already inspected in the initial run; expanded results are follow-up findings. No fresh independent holdout has been obtained. Dataset anonymization, retained duplicates and absent account identifiers limit interpretation and independence analysis.

The search is bounded and staged. The diagnostic studies use fixed K=4, three seeds and a small dimensionality grid; they do not establish universal numerical robustness. Density fit and covariance conditioning do not directly optimize fraud utility. Bootstrap intervals omit temporal dependence, refitting and prevalence uncertainty. Batch review capacity also differs from a streaming queue.

Phase 2 will train an undercomplete or denoising autoencoder using the frozen legitimate training membership and preprocessing. Architecture, early stopping and thresholds must use validation. The model-independent metrics remain the same. Any hybrid needs a rationale and ablation evidence. Continued reuse of this test partition must be disclosed.

\\section{Conclusion}
The expanded Phase 1 study combines train-only GMM selection with explicit validation threshold policies, reproducible split membership and documented sensitivity studies. Operational confusion counts and review workload make the trade-offs visible. Validation-only robustness and post-freeze error analysis clarify limitations without treating convergence or low density as proof of fraud. The completed classical pipeline provides the frozen interfaces needed for Phase 2.

'''+s[b:]
    # Remove an old duplicated PR float if still present before the new results section.
    cut=s.index('\\section{Results and Operational Trade-offs}')
    head=re.sub(r'\\begin\{figure\}\[t\]\n\\centering\\includegraphics\[width=\\columnwidth\]\{figures/pr_curves.png\}.*?\\end\{figure\}', '',s[:cut],flags=re.S)
    s=head+s[cut:]
    tex.write_text(s)
    put('guides/VIVA_GUIDE.md',f'''# Team T085: viva preparation

All members should understand the complete protocol as well as their assigned role. Role 1 Gauri: data/splits. Role 2 Farhana: preprocessing. Role 3 Aaditya: GMM. Role 4 Bhumi: metrics. Role 5 Anshvardhan: robustness, error analysis and integration narrative.

## Core explanations

1. **Why novelty detection?** Normal records are plentiful, while fraud patterns can change. The assumption is that useful frauds deviate from legitimate behavior. Unusual legitimate transactions and normal-looking fraud limit this assumption.
2. **Is fitting label-free?** Labels filter the known legitimate training population. No labels enter model inputs or loss. Call it clean-normal novelty detection, not a method with no label access.
3. **What is a GMM?** A weighted sum of Gaussian densities with weights summing to one. The anomaly score is negative log density. Density is not a fraud probability and may exceed one.
4. **How does EM fit it?** The E-step calculates soft component responsibilities. The M-step updates weighted means, covariance matrices and mixing weights. Multiple starts reduce local-optimum risk without guaranteeing a global optimum.
5. **What do covariance types mean?** Full models correlations per component, diagonal assumes component-wise axis-aligned covariance, tied shares a full matrix, and spherical uses a scalar variance per component.
6. **What are AIC and BIC?** AIC=2p-2 log L; BIC=p log n-2 log L. Compare candidates on identical observations in the same feature space. Negative criteria are possible for continuous densities. BIC is the fixed primary selection rule.
7. **What was searched?** Twenty-eight structural candidates cross seven component counts and four covariance types. Predetermined one-factor trials change regularization, starts and iteration caps on the structural winner. This is bounded staged tuning rather than an exhaustive grid.
8. **What was selected?** {int(selected.components)} {selected.covariance}-covariance components, reg_covar={selected.reg_covar:g}, n_init={int(selected.n_init)}, max_iter={int(selected.max_iter)}. Training BIC={selected.bic:.1f}. Report the grid-boundary limitation if applicable.
9. **Why regularize covariance?** Adding positive diagonal variance reduces singularity risk. The selected largest condition number is {cond:,.1f}. Convergence alone does not establish good conditioning, and good conditioning alone does not establish fraud utility.
10. **How are splits frozen?** Original source row IDs, all three chronological partitions and legitimate training membership are saved in split_indices.npz. Loading checks dataset SHA, fractions, complete/nonoverlapping membership, chronology and normal labels. Equal timestamps stay together.
11. **What is preprocessing?** V1-V28 plus log1p(Amount), fitted legitimate-training StandardScaler, and optional training-fitted PCA for diagnostic experiments. Time defines ordering and EDA. Class never enters X. Save and reuse the full pipeline.
12. **Why not accuracy?** With rare fraud, majority classification can have high accuracy while catching no fraud. Precision, recall, FPR, AP and workload address the review problem. Accuracy is not an evaluation output.
13. **AP versus PR AUC?** AP sums recall increments weighted by precision. Trapezoidal PR AUC joins curve points linearly. GMM AP={test.loc['gmm','average_precision']:.4f}; trapezoidal area={test.loc['gmm','pr_auc_trapezoidal']:.4f}. Use each split's own prevalence for the no-skill reference.
14. **How do threshold ties work?** At alpha, allow floor(alpha*n_normal_validation) false positives. Use the next descending normal score as cutoff and flag strictly greater scores. Ties make validation calibration conservative. Test FPR can differ.
15. **Explain the 0.5% FPR policy.** Frozen cutoff={point.threshold:.4f}. Test TP={int(point.tp)}, FP={int(point.fp)}, FN={int(point.fn)}, TN={int(point.tn)}. Recall={pct(point.recall)}%, precision={pct(point.precision)}%, actual FPR={pct(point.fpr,3)}% and alerts={int(point.flagged)}.
16. **Explain capacity.** Top0.5% reviews {int(cap.k)} scored test records and catches {int(cap.tp)} frauds at {pct(cap.precision_at_k)}% precision. Top100 catches {int(count.tp)}. Queue membership uses scores with stable chronological ties, never labels. Capacity is distinct from a frozen threshold.
17. **What robustness was completed?** {len(rob)} validation-only fits compare four dimensions, four covariance types, three seeds and nested training fractions at fixed K=4. Sample-size studies keep full-training preprocessing fixed. Raw AIC/BIC cannot rank changed dimensions or fitting populations.
18. **What do seed metrics show?** AP mean/sample SD measure metric variation. Spearman correlations measure ranking agreement. Per-row raw score SD measures scale-sensitive score variation. Three seeds and 57 validation frauds limit conclusions.
19. **What does error analysis establish?** It describes missed frauds, high-score legitimate alerts, group amounts/times and standardized feature deviations after the freeze. It cannot identify causes from anonymized features and does not retune the model. At the default error policy there are {point.fp/point.tp:.2f} false positives per caught fraud.
20. **Is test evaluation independent?** The initial run already inspected this test partition. The expansion keeps train-only selection and validation-only diagnostics but reports test reuse. It must not be described as a fresh untouched holdout.
21. **What does the bootstrap interval mean?** GMM AP interval={test.loc['gmm','ap_ci_lower']:.4f}-{test.loc['gmm','ap_ci_upper']:.4f} from500 stratified resamples. It conditions on observed prevalence and omits refitting and temporal dependence. It is not a paired superiority test.
22. **What is left for Phase 2?** Fit an undercomplete/denoising autoencoder on exactly the same normal training membership and preprocessing. Select neural settings on validation, reuse the metrics harness, disclose test reuse and justify any hybrid using ablation evidence.

## Demonstration sequence

Run `python -m unittest discover -s tests -v`, inspect split membership and preprocessing, explain the selected grid row, recompute the default confusion metrics by hand and open notebooks 04-06. Run `python scripts/verify_outputs.py` to check the actual score and metric contracts. Individual members must prepare their own explanations and record real review/contribution evidence.
''')
    print('Updated result narratives, README, robustness analysis, error analysis, viva guide and IEEE source.')

if __name__=='__main__':main()
