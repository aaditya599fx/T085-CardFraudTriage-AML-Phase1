"""Train-only AIC/BIC model selection, staged optimization and numerical diagnostics."""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture


def covariance_diagnostics(model):
    cov = model.covariances_
    if model.covariance_type == 'full':
        eigenvalues = [np.linalg.eigvalsh(c) for c in cov]
    elif model.covariance_type == 'tied':
        eigenvalues = [np.linalg.eigvalsh(cov)]
    elif model.covariance_type == 'diag':
        eigenvalues = list(cov)
    else:
        eigenvalues = [np.array([v]) for v in cov]
    minima = [float(e.min()) for e in eigenvalues]
    maximum_condition = max(float(e.max()/e.min()) if e.min() > 0 else np.inf for e in eigenvalues)
    return dict(min_cov_eigenvalue=min(minima), max_cov_condition=maximum_condition,
                ill_conditioned=bool(maximum_condition > 1e6))


def fit_candidate(x, *, k, covariance, reg_covar, n_init, max_iter, seed):
    started = time.perf_counter()
    record = dict(components=k, covariance=covariance, reg_covar=reg_covar,
                  n_init=n_init, max_iter=max_iter, seed=seed, fit_rows=len(x), dimensions=x.shape[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        try:
            model = GaussianMixture(n_components=k, covariance_type=covariance,
                reg_covar=reg_covar, n_init=n_init, max_iter=max_iter, random_state=seed).fit(x)
            record.update(aic=float(model.aic(x)), bic=float(model.bic(x)),
                converged=bool(model.converged_), iterations=int(model.n_iter_),
                min_weight=float(model.weights_.min()), **covariance_diagnostics(model), error='')
        except (ValueError, np.linalg.LinAlgError) as exc:
            model = None
            record.update(aic=np.nan, bic=np.nan, converged=False, iterations=0,
                min_weight=np.nan, min_cov_eigenvalue=np.nan, max_cov_condition=np.nan,
                ill_conditioned=False, error=str(exc))
    record['warnings'] = '; '.join(str(w.message) for w in caught)
    record['seconds'] = time.perf_counter()-started
    return model, record


def fit_grid(x, config, seed):
    """Search structure, then change one optimizer setting at a time on the BIC winner.

    All AIC/BIC values share the same normal training observations and feature space.
    No validation or test arrays are accepted by this selection function.
    """
    records, best_model, best_value = [], None, np.inf
    best_index = None
    def consider(params, stage):
        nonlocal best_model, best_value, best_index
        model, record = fit_candidate(x, seed=seed, **params)
        record['stage'] = stage
        idx = len(records)
        records.append(record)
        if model is not None and record['converged'] and np.isfinite(record[config['selection']]):
            if record[config['selection']] < best_value:
                best_model, best_value, best_index = model, record[config['selection']], idx
        print(f'GMM {stage} {params}: BIC={record["bic"]:.1f}, converged={record["converged"]}', flush=True)
        return record
    for covariance in config['covariance_types']:
        for k in config['components']:
            consider(dict(k=k, covariance=covariance, reg_covar=config['reg_covar'],
                          n_init=config['n_init'], max_iter=config['max_iter']), 'structure')
    if best_model is None:
        raise RuntimeError('No converged structure candidate; inspect the diagnostics.')
    anchor = dict(k=best_model.n_components, covariance=best_model.covariance_type,
                  reg_covar=config['reg_covar'], n_init=config['n_init'], max_iter=config['max_iter'])
    # Predetermined one-factor tuning, not a claim of an exhaustive Cartesian search.
    tuning = config.get('tuning', {})
    seen = {tuple(anchor.items())}
    for setting in ['reg_covar', 'n_init', 'max_iter']:
        for value in tuning.get(setting, []):
            params = {**anchor, setting:value}
            key = tuple(params.items())
            if key not in seen:
                seen.add(key)
                consider(params, 'optimizer')
    table = pd.DataFrame(records)
    table['selected'] = table.index == best_index
    return best_model, table


def score(model, x):
    return -model.score_samples(x)


def plot_selection(grid, directory):
    """Plot structural curves separately from the one-factor optimizer candidates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    structural=grid[grid.stage=='structure']
    chosen=grid[grid.selected].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for cov,rows in structural.groupby('covariance'):
        for ax,criterion in zip(axes,['aic','bic']):
            ax.plot(rows.components,rows[criterion],marker='o',label=cov)
            ax.set(xlabel='Components',ylabel=criterion.upper())
    for ax,criterion in zip(axes,['aic','bic']):
        ax.scatter([chosen.components],[chosen[criterion]],marker='*',s=100,color='black',label='Selected optimizer fit',zorder=5)
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(directory/'gmm_selection.png');plt.close(fig)
