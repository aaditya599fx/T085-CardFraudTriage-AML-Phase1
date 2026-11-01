"""Predetermined sensitivity studies using validation only; test data are never accepted."""
import itertools
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .features import preprocess
from .gmm_model import fit_candidate, score
from .metrics import ranking, fpr_threshold, at_threshold


def run_robustness(train_legitimate, validation, config, out):
    started = time.perf_counter()
    y = validation.Class.to_numpy()
    dimensions = config['dimensions']
    arrays = {}
    for dim in dimensions:
        pca_dim = None if dim == 29 else dim
        xt, pipeline = preprocess(train_legitimate, pca_dimensions=pca_dim)
        xv, _ = preprocess(validation, pipeline=pipeline)
        arrays[dim] = (xt, xv)
        joblib.dump(pipeline, out/f'models/preprocessing_{dim}d.joblib')
    records, seed_scores = [], {}
    anchor_seed = config['seeds'][1] if len(config['seeds']) > 1 else config['seeds'][0]
    permutation = np.random.default_rng(anchor_seed).permutation(len(train_legitimate))
    def run(dim, cov, seed, fraction, study):
        xt, xv = arrays[dim]
        subset = xt if fraction == 1 else xt[permutation[:max(config['components'],int(len(xt)*fraction))]]
        model, row = fit_candidate(subset, k=config['components'], covariance=cov,
            reg_covar=config['reg_covar'], n_init=config['n_init'],
            max_iter=config['max_iter'], seed=seed)
        row.update(study=study, train_fraction=fraction)
        if model is not None:
            ss = score(model,xv)
            if np.isfinite(ss).all():
                row.update({f'validation_{k}':float(v) for k,v in ranking(y,ss).items()})
                point = at_threshold(y,ss,fpr_threshold(y,ss,.005))
                row['validation_recall_at_005_fpr'] = point['recall']
                if dim == 29 and fraction == 1:
                    seed_scores[cov,seed] = ss
        records.append(row)
        pd.DataFrame(records).to_csv(out/'tables/robustness_trials.csv',index=False)
        print(f'Robustness {study}: d={dim} cov={cov} seed={seed} fraction={fraction}, AP={row.get("validation_average_precision",np.nan):.4f}',flush=True)
    for dim,cov in itertools.product(dimensions,config['covariance_types']):
        run(dim,cov,anchor_seed,1.,'dimensions_covariance')
    for cov,seed in itertools.product(config['covariance_types'],config['seeds']):
        if seed != anchor_seed:
            run(29,cov,seed,1.,'seed')
    for fraction in config['train_fractions']:
        if fraction != 1:
            run(29,'full',anchor_seed,fraction,'training_size')
    trials = pd.DataFrame(records)
    seeds = trials[(trials.dimensions == 29) & (trials.train_fraction == 1)]
    summary = seeds.groupby('covariance').agg(
        runs=('seed','size'),converged_runs=('converged','sum'),
        validation_ap_mean=('validation_average_precision','mean'),
        validation_ap_std=('validation_average_precision','std'),
        validation_ap_min=('validation_average_precision','min'),
        validation_ap_max=('validation_average_precision','max'),
        maximum_condition=('max_cov_condition','max')).reset_index()
    summary.to_csv(out/'tables/seed_summary.csv',index=False)
    corr_rows, dispersion = [], []
    for cov in config['covariance_types']:
        available = [seed for seed in config['seeds'] if (cov,seed) in seed_scores]
        for a,b in itertools.combinations(available,2):
            corr_rows.append(dict(covariance=cov,seed_a=a,seed_b=b,
                validation_spearman=float(spearmanr(seed_scores[cov,a],seed_scores[cov,b]).statistic)))
        stack = np.array([seed_scores[cov,s] for s in available])
        if len(stack):
            sd = stack.std(axis=0,ddof=1) if len(stack)>1 else np.zeros(stack.shape[1])
            for rowid,mean,std in zip(validation.index,stack.mean(axis=0),sd):
                dispersion.append(dict(covariance=cov,source_row=int(rowid),score_mean=mean,score_std=std))
    pd.DataFrame(corr_rows).to_csv(out/'tables/seed_rank_correlations.csv',index=False)
    pd.DataFrame(dispersion).to_csv(out/'tables/seed_score_dispersion.csv.gz',index=False,compression='gzip')
    fig,axes = plt.subplots(1,2,figsize=(10,3.8))
    dimrows = trials[trials.study == 'dimensions_covariance']
    for cov,rows in dimrows.groupby('covariance'):
        axes[0].plot(rows.dimensions,rows.validation_average_precision,marker='o',label=cov)
        axes[1].semilogy(rows.dimensions,rows.max_cov_condition,marker='o',label=cov)
    axes[0].set(xlabel='Input dimensions',ylabel='Validation average precision')
    axes[1].set(xlabel='Input dimensions',ylabel='Maximum covariance condition number')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'figures/dimensionality_stability.png');plt.close(fig)
    fig,ax = plt.subplots(figsize=(6,3.8))
    ax.bar(summary.covariance,summary.validation_ap_mean,yerr=summary.validation_ap_std,capsize=5,color='#147d92')
    ax.set(xlabel='Covariance (K=4, full feature space)',ylabel='Validation AP: mean ± SD across seeds')
    fig.tight_layout();fig.savefig(out/'figures/seed_stability.png');plt.close(fig)
    with (out/'robustness_metadata.json').open('w') as f:
        import json
        json.dump(dict(config=config,seconds=time.perf_counter()-started,trials=len(trials),
          evaluation_split='validation only',selection='Diagnostic study; no ranking across dimensional spaces by AIC/BIC.',
          training_size_policy='Nested seeded subsets; full-training preprocessing held fixed to isolate GMM sample-size effects.'),f,indent=2)
    return trials
