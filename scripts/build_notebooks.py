"""Build the seven Phase 1 notebooks from reproducible source; execute them with Jupyter."""
from pathlib import Path
import nbformat as n
R=Path(__file__).resolve().parents[1]
setup='''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
assert (ROOT/'src').exists(), 'Start Jupyter in the project folder.'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
TABLES=ROOT/'results/tables'
FIGURES=ROOT/'results/figures'
def table(name): return pd.read_csv(TABLES/name)
def figure(name): display(Image(filename=str(FIGURES/name)))
'''
def md(s):return n.v4.new_markdown_cell(s.replace('\\n','\n'))
def code(s):return n.v4.new_code_cell(s)
def save(name,cells):
 nb=n.v4.new_notebook(cells=[md('# Team T085 · Section E · Project 26\nPhase 1: Machine Learning\n\nThe expanded run reuses the previously inspected test partition. Training-only selection and validation-only robustness are preserved; this is not a fresh independent holdout.'),code(setup)]+cells)
 nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
 n.write(nb,R/'notebooks'/name)
load='''from src.data import load_raw, load_split_indices
import yaml
cfg=yaml.safe_load((ROOT/'configs/config.yaml').read_text())
path=ROOT/'data/raw/creditcard.csv'
raw=load_raw(path)
parts=load_split_indices(raw,ROOT/'data/processed/split_indices.npz',path,cfg['split_fractions'])
'''
save('01_literature_and_eda.ipynb',[
 md('## Data and literature\nOwner: Role 1. Read reports/literature_review.md for EM, isolation and precision-recall background. Models fit known legitimate training records; labels filter that population but never enter the model features. Training EDA covers quality, base rates, Amount, Time and V1-V28.'),
 code(load+"display(table('split_summary.csv').drop(columns='index_sha256'))\nprint(json.loads((ROOT/'data/processed/split_manifest.json').read_text())['quality'])\nprint('Loaded exact frozen row membership from split_indices.npz.')"),
 code("train=parts['train']\ndisplay(train.describe().round(3))\nassert train.Time.max()<parts['validation'].Time.min()\nassert parts['validation'].Time.max()<parts['test'].Time.min()\nwith np.load(ROOT/'data/processed/split_indices.npz') as z:\n    np.testing.assert_array_equal(z['train_legitimate'],train.index[train.Class==0])"),
 code("figure('eda_class_imbalance.png')\nfigure('eda_amount_time.png')\nfigure('eda_correlation.png')\nfigure('eda_v_distributions.png')"),
 md('## Quality interpretation\nThe CSV has no missing/non-finite cells and 1,081 exact duplicates. They are retained because transaction identifiers are absent. Equal Time values stay together, preventing exact duplicates spanning splits. V distribution panels show the central 98% training range per feature; extreme values remain in model fitting. The released PCA features cannot be independently refitted.')])
save('02_feature_engineering.ipynb',[
 md('## Reusable preprocessing\nOwner: Role 2. Inputs are V1-V28 and log1p(Amount). Time defines chronology. A full raw-frame pipeline fits StandardScaler and optional PCA exclusively on legitimate training rows.'),
 code(load+"from src.features import prepare,preprocess,engineering\nnormal=parts['train'][parts['train'].Class==0]\nxt,pipeline=preprocess(normal)\nmean_before=pipeline.named_steps['scaler'].mean_.copy()\nxv,_=preprocess(parts['validation'],pipeline=pipeline)\nnp.testing.assert_array_equal(mean_before,pipeline.named_steps['scaler'].mean_)\nnp.testing.assert_allclose(mean_before,engineering(normal).mean(axis=0))\nprint('Training/validation shapes:',xt.shape,xv.shape)"),
 code("from src.data import FEATURES\ndisplay(pd.DataFrame({'feature':FEATURES,'mean':mean_before,'scale':pipeline.named_steps['scaler'].scale_}))\nfor d in [5,10,20]:\n    z,pp=preprocess(normal,pca_dimensions=d)\n    print(d,'components; retained training variance:',pp.named_steps['pca'].explained_variance_ratio_.sum())\n    assert z.shape==(len(normal),d)"),
 md('## Leakage checklist\nReuse fitted transformations, never fit validation/test, preserve feature order, and never include Class in X. Study pipelines are saved locally in results/models; the Git repository documents regeneration. Provider PCA provenance remains unaudited.')])
save('03_gmm_baseline.ipynb',[
 md(r'## GMM selection\nOwner: Role 3. $p(x)=\sum_k\pi_k\mathcal N(x;\mu_k,\Sigma_k)$ and $s(x)=-\log p(x)$. EM computes soft responsibilities and weighted parameter updates. Training AIC/BIC share identical fitting observations and feature space. A staged optimizer search follows the 28 structural fits. Neither validation nor test labels choose the model.'),
 code("grid=table('gmm_grid.csv')\ndisplay(grid.drop(columns=['warnings','error']))\nselected=grid[grid.selected]\ndisplay(selected)\nprint('AIC winner:');display(grid[grid.converged].sort_values('aic').head(1))\nprint('Warnings and failures:');display(grid[(grid.warnings.fillna('')!='') | (~grid.converged)])\nfigure('gmm_selection.png')"),
 code(load+"import joblib\nfrom src.gmm_model import score\nfrom src.features import preprocess\nmodel_path=ROOT/'results/models/gmm.joblib'\nif model_path.exists():\n    model=joblib.load(model_path)\n    pipeline=joblib.load(ROOT/'results/models/preprocessing.joblib')\n    xv,_=preprocess(parts['validation'],pipeline=pipeline)\n    computed=score(model,xv)\n    saved=np.load(ROOT/'results/scores/gmm_validation.npz')\n    np.testing.assert_allclose(computed,saved['score'])\n    np.testing.assert_array_equal(parts['validation'].index,saved['source_row'])\n    print('Validation scores and source IDs reproduced from saved model/pipeline.')\nelse:\n    print('Run python run_all.py to regenerate fitted model binaries.')"),
 code("# Optional training, disabled during result inspection.\nRETRAIN=False\nif RETRAIN:\n    from src.features import prepare\n    from src.gmm_model import fit_grid\n    from threadpoolctl import threadpool_limits\n    xtrain,_,_=prepare(parts)\n    with threadpool_limits(limits=2):\n        model,grid=fit_grid(xtrain,cfg['gmm'],cfg['seed'])"),
 md('## Interpretation\nReport convergence separately from conditioning and fraud ranking. The largest tested component count is only a bounded search endpoint. AIC/BIC do not select across different PCA dimensions or sample sizes. Continuous density and component membership are not fraud posterior probabilities.')])
save('03b_classical_baselines.ipynb',[
 md('## Fixed classical comparators\nAll models receive the same normal-training observations and transformed features. Isolation Forest: 200 trees, 256 sampled records/tree. PCA: 15 components with full SVD and reconstruction MSE. Approximate OC-SVM: 128 Nyström RBF features (gamma=1/29), scaling, SGDOneClassSVM(nu=.01). Baselines remain fixed; exact kernel OC-SVM and LOF are outside this implementation.'),
 code("metrics=table('ranking_metrics.csv')\ndisplay(metrics[metrics.split=='validation'].drop(columns=['ap_ci_lower','ap_ci_upper']))"),
 code("manifest=json.loads((ROOT/'data/processed/split_manifest.json').read_text())\nfor name in ['gmm','isolation_forest','pca','approx_ocsvm']:\n    for part in ['validation','test']:\n        saved=np.load(ROOT/f'results/scores/{name}_{part}.npz')\n        with np.load(ROOT/'data/processed/split_indices.npz') as ids:\n            np.testing.assert_array_equal(saved['source_row'],ids[part])\n        assert np.isfinite(saved['score']).all()\nprint('All eight model/split score arrays match the frozen row membership.')"),
 md('## Negative results\nKeep weak comparator configurations in the report. They describe those fixed settings rather than establish that the entire model family fails. Original baseline settings are not retuned after test inspection.')])
save('04_robustness_and_error_analysis.ipynb',[
 md('## Validation-only sensitivity\nOwner: Role 5, with Role 2 for PCA and Role 3 for GMM configurations. The robustness function accepts legitimate training and mixed validation only. Fixed K=4 isolates dimension/covariance, random seeds and model sample-size effects. Test error analysis is a separate post-freeze descriptive step.'),
 code("trials=table('robustness_trials.csv')\ndisplay(trials.drop(columns=['warnings','error']))\nprint('Diagnostic fits:',len(trials),'Nonconverged/failed:',int((~trials.converged).sum()))\nfigure('dimensionality_stability.png')\nfigure('seed_stability.png')"),
 code("display(table('seed_summary.csv'))\ndisplay(table('seed_rank_correlations.csv'))\ndisplay(trials[trials.study=='training_size'])\nprint('Per-observation raw-score SD:')\ndisp=table('seed_score_dispersion.csv.gz')\ndisplay(disp.groupby('covariance').score_std.describe())"),
 code("display(table('error_group_summary.csv'))\ndisplay(table('missed_frauds.csv').head(10))\ndisplay(table('top_false_positives.csv'))\nfigure('error_analysis.png')"),
 code("display(Markdown((ROOT/'reports/robustness_analysis.md').read_text()))\ndisplay(Markdown((ROOT/'reports/error_analysis.md').read_text()))"),
 md('## Interpretation limits\nDo not compare raw AIC/BIC across dimensions or fitting populations. Spherical covariance conditions are one by construction. Training-size studies hold full-training preprocessing fixed and therefore isolate GMM fitting size. Three seeds and 57 validation frauds cannot establish universal stability. Error features are descriptive, not causal explanations.')])
save('05_evaluation.ipynb',[
 md('## Model-independent metrics and frozen policies\nOwner: Role 4. Validate strict score>threshold policies at 0.1%,0.5%,1%,5% FPR, then apply frozen cutoffs to test. Capacity ranking separately reviews top fractions or top100. The test set has already been inspected in the initial run; disclose reuse.'),
 code("display(table('ranking_metrics.csv').round(6))\nfigure('pr_curves.png')"),
 code("ops=table('operating_points.csv')\ndisplay(ops[(ops.model=='gmm')&(ops.split=='test')].round(6))\nfigure('gmm_confusion_matrices.png')\nfigure('threshold_tradeoffs.png')"),
 code("display(table('review_capacity.csv').query(\"split=='test'\").round(6))\ndisplay(table('review_workload.csv').query(\"model=='gmm' and split=='test'\").round(6))\nassert (ops[ops.split=='validation'].fpr<=ops[ops.split=='validation'].fpr_budget+1e-12).all()\nrow=ops[(ops.model=='gmm')&(ops.split=='test')&np.isclose(ops.fpr_budget,.005)].iloc[0]\nassert np.isclose(row.recall,row.tp/(row.tp+row.fn))\nassert np.isclose(row.precision,row.tp/(row.tp+row.fp))\nassert np.isclose(row.fpr,row.fp/(row.fp+row.tn))\nprint('Confusion-count identities and validation FPR constraints verified.')"),
 code("from src.metrics import evaluate_scores, load_thresholds\nwith np.load(ROOT/'data/processed/split_indices.npz') as ids:\n    order=ids['test']\nfrom src.data import load_raw\nraw=load_raw(ROOT/'data/raw/creditcard.csv')\ny=raw.loc[order,'Class'].to_numpy()\ns=np.load(ROOT/'results/scores/gmm_test.npz')['score']\nresult=evaluate_scores(y,s,thresholds=[load_thresholds(ROOT/'results/tables/thresholds_frozen_on_validation.csv').query(\"model=='gmm' and fpr_budget==0.005\").threshold.iloc[0]],review_fractions=[.005],review_counts=[100])\nprint(result)\nassert result['operating_points'][0]['tp']==row.tp"),
 md('## Limits\nAP differs from trapezoidal PR AUC. Actual test FPR may differ from the validation budget. Undefined workload ratios when TP=0 remain blank with an explicit flag. Stratified AP intervals omit temporal dependence, refitting and prevalence uncertainty. Do not retune using these outcomes.')])
save('06_phase1_summary.ipynb',[
 md('## Assembled Phase 1 narrative\nOwner: Role 5 integrates reviewed contributions from all roles. This notebook summarizes the completed classical pipeline and links its frozen interfaces to Phase 2.'),
 code("display(Markdown((ROOT/'README.md').read_text()))"),
 code("display(table('split_summary.csv').drop(columns='index_sha256'))\ndisplay(table('gmm_grid.csv').query('selected'))\ndisplay(table('ranking_metrics.csv').query(\"split=='test'\"))\ndisplay(table('seed_summary.csv'))"),
 code("figure('gmm_selection.png')\nfigure('pr_curves.png')\nfigure('dimensionality_stability.png')\nfigure('error_analysis.png')"),
 md('## Phase 2 placeholder\nTrain an undercomplete or denoising autoencoder only on the saved legitimate training rows. Reuse the exact full preprocessing pipeline and model-independent evaluation harness. Use validation for architecture, early stopping and thresholds. Disclose the already inspected test partition. A hybrid requires a rationale and ablation evidence; no neural or hybrid training is claimed in Phase 1.')])
print('Built seven Phase 1 notebooks.')
