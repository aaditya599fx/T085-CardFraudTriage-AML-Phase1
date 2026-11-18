"""Evaluate the frozen historical comparison and run validation-only AE ablations."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from src.data import load_raw, load_split_indices
from src.features import preprocess
from src.metrics import (
    at_capacity, at_review_count, at_threshold, fpr_threshold, ranking,
)

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / 'data/processed/split_indices.npz'
PHASE2 = ROOT / 'results/phase2'
BUDGETS = (0.001, 0.005, 0.01, 0.05)
CAPACITIES = (0.001, 0.005, 0.01)
VARIANTS = {
    'Denoising, width 8': (8, 0.05),
    'Clean, width 8': (8, 0.0),
    'Denoising, width 4': (4, 0.05),
    'Denoising, width 16': (16, 0.05),
}
SEEDS = (13, 26, 42)
MAX_EPOCHS = 35
PATIENCE = 5
BATCH_SIZE = 512


def read_frozen_scores(path, expected_rows):
    with np.load(path, allow_pickle=False) as saved:
        scores = saved['score']
        if not np.array_equal(saved['source_row'], expected_rows):
            raise ValueError(f'Frozen scores do not match the split order: {path}')
    if scores.shape != (len(expected_rows),) or not np.isfinite(scores).all():
        raise ValueError(f'Expected one finite score per transaction: {path}')
    return scores


def historical_comparison(parts, out):
    labels = {split: parts[split].Class.to_numpy() for split in ('validation', 'test')}
    ranks, operating, capacities, pr_points = [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for model, source, color in (
        ('GMM', ROOT / 'results/scores', '#0A6978'),
        ('Autoencoder', PHASE2, '#B65625'),
    ):
        scores = {
            split: read_frozen_scores(source / f'{model.lower()}_{split}.npz', parts[split].index)
            for split in labels
        }
        cutoffs = {b: fpr_threshold(labels['validation'], scores['validation'], b) for b in BUDGETS}
        for ax, (split, y) in zip(axes, labels.items()):
            score = scores[split]
            record = {'model': model, 'split': split}
            ranks.append({**record, **ranking(y, score)})
            for budget, cutoff in cutoffs.items():
                operating.append({**record, 'fpr_budget': budget, 'threshold_hex': cutoff.hex(),
                                  **at_threshold(y, score, cutoff)})
            for fraction in CAPACITIES:
                capacities.append({**record, 'policy': 'fraction', **at_capacity(y, score, fraction)})
            capacities.append({**record, 'policy': 'top 100', **at_review_count(y, score, 100)})
            precision, recall, _ = precision_recall_curve(y, score)
            pr_points.append(pd.DataFrame({**record, 'recall': recall, 'precision': precision}))
            ax.step(recall[::-1], precision[::-1], where='post', color=color, label=model)
    for ax, (split, y) in zip(axes, labels.items()):
        ax.axhline(y.mean(), color='#69747A', linestyle=':', label='Fraud prevalence')
        ax.set(title='Historical test (reused)' if split == 'test' else 'Validation',
               xlabel='Recall', xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc='upper right')
    axes[0].set_ylabel('Precision')
    fig.tight_layout()
    fig.savefig(out / 'precision_recall.png', dpi=170)
    plt.close(fig)
    for name, records in (('ranking', ranks), ('operating_points', operating), ('capacity', capacities)):
        pd.DataFrame(records).to_csv(out / f'{name}.csv', index=False)
    pd.concat(pr_points, ignore_index=True).to_csv(
        out / 'precision_recall_points.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 1},
    )


def reconstruction_error(model, clean):
    return np.mean(np.square(clean - model.predict(clean)), axis=1)


def fit_autoencoder(train, stopping, width, noise_std, seed, max_epochs=MAX_EPOCHS):
    model = MLPRegressor(hidden_layer_sizes=(16, width, 16), activation='relu', solver='adam',
                         batch_size=BATCH_SIZE, alpha=1e-4, learning_rate_init=1e-3,
                         shuffle=False, early_stopping=False, random_state=seed)
    shuffle_rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    history = []
    best_loss, best_epoch = np.inf, 0
    for epoch in range(1, max_epochs + 1):
        clean = train[shuffle_rng.permutation(len(train))]
        noise = noise_rng.standard_normal(clean.shape, dtype=clean.dtype)
        model.partial_fit(clean + noise_std * noise, clean)
        reserve_mse = float(reconstruction_error(model, stopping).mean())
        history.append({'epoch': epoch, 'training_objective': float(model.loss_),
                        'reserve_mse': reserve_mse})
        if reserve_mse < best_loss:
            best_loss, best_epoch = reserve_mse, epoch
            best_weights = ([w.copy() for w in model.coefs_], [b.copy() for b in model.intercepts_])
        if epoch - best_epoch >= PATIENCE:
            break
    model.coefs_, model.intercepts_ = best_weights
    return model, history, best_epoch


def validation_ablations(train_part, validation, out):
    legitimate = train_part.loc[train_part.Class.eq(0)]
    time = legitimate.Time.to_numpy()
    cut = np.searchsorted(time, time[int(0.9 * len(time))], side='left')
    if not 0 < cut < len(time):
        raise ValueError('The legitimate training split needs distinct times for the stopping reserve.')
    fit, pipeline = preprocess(legitimate.iloc[:cut])
    fit = fit.astype(np.float32)
    reserve = pipeline.transform(legitimate.iloc[cut:]).astype(np.float32)
    x_val = pipeline.transform(validation).astype(np.float32)
    y_val = validation.Class.to_numpy()
    rows, curves = [], []
    for variant, (width, noise) in VARIANTS.items():
        for seed in SEEDS:
            model, history, best = fit_autoencoder(fit, reserve, width, noise, seed)
            score = reconstruction_error(model, x_val)
            operating = at_threshold(y_val, score, fpr_threshold(y_val, score, 0.005))
            tag = {'variant': variant, 'seed': seed}
            rows.append({**tag, 'width': width, 'noise_std': noise, 'epochs': len(history),
                         'best_epoch': best, 'best_reserve_mse': history[best - 1]['reserve_mse'],
                         **ranking(y_val, score), 'recall_at_005_fpr': operating['recall'],
                         'precision_at_005_fpr': operating['precision']})
            curves.extend({**tag, **point} for point in history)
            print(f'{variant}, seed {seed}: validation AP={rows[-1]["average_precision"]:.4f}', flush=True)
    trials = pd.DataFrame(rows)
    curves = pd.DataFrame(curves)
    trials.to_csv(out / 'ablation_trials.csv', index=False)
    curves.to_csv(out / 'training_curves.csv', index=False)
    summary = trials.groupby('variant', sort=False).agg(
        width=('width', 'first'), noise_std=('noise_std', 'first'), runs=('seed', 'size'),
        mean_ap=('average_precision', 'mean'), sd_ap=('average_precision', 'std'),
        mean_pr_auc=('pr_auc_trapezoidal', 'mean'), mean_recall=('recall_at_005_fpr', 'mean'),
        mean_precision=('precision_at_005_fpr', 'mean'),
    )
    summary.to_csv(out / 'ablation_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    for ax, (variant, group) in zip(axes.flat, curves.groupby('variant', sort=False)):
        for seed, trial in group.groupby('seed'):
            ax.plot(trial.epoch, trial.reserve_mse, label=f'Reserve MSE, seed {seed}')
        reference = group[group.seed.eq(26)]
        ax.plot(reference.epoch, reference.training_objective, linestyle='--', color='#4F5457',
                label='Training objective, seed 26')
        ax.set(title=variant, xlabel='Epoch', ylabel='Loss or MSE')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'training_curves.png', dpi=165)
    plt.close(fig)
    return {'fit_legitimate_rows': len(fit), 'stopping_legitimate_rows': len(reserve),
            'fit_time_end': float(time[cut - 1]), 'reserve_time_start': float(time[cut]),
            'variants': VARIANTS, 'seeds': SEEDS, 'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE, 'batch_size': BATCH_SIZE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/raw/creditcard.csv')
    parser.add_argument('--out', type=Path, default=PHASE2 / 'study')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parts = load_split_indices(load_raw(args.data), SPLIT_PATH, args.data, (0.6, 0.2, 0.2))
    # Inner training uses legitimate rows and mixed validation. Test enters the frozen-score audit only.
    with threadpool_limits(limits=2):
        ablation = validation_ablations(parts['train'], parts['validation'], args.out)
        historical_comparison(parts, args.out)
    with np.load(SPLIT_PATH, allow_pickle=False) as saved:
        dataset_hash = saved['dataset_sha256'].item()
    inputs = [SPLIT_PATH, *(PHASE2 / f'autoencoder_{s}.npz' for s in ('validation', 'test')),
              *(ROOT / f'results/scores/gmm_{s}.npz' for s in ('validation', 'test'))]
    metadata = {
        'dataset_sha256': dataset_hash, 'historical_test_previously_inspected': True,
        'ablation_mixed_split': 'validation', 'validation_fpr_budgets': BUDGETS,
        'capacity_fractions': CAPACITIES, 'ablation': ablation,
        'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'ablation_runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__,
                        'scikit_learn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
    }
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
