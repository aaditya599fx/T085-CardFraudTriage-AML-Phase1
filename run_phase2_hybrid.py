"""Fuse the frozen Phase 1 GMM and Phase 2 autoencoder scores for fraud triage."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from run_phase2_study import BUDGETS, CAPACITIES, PHASE2, ROOT, SPLIT_PATH, read_frozen_scores
from src.data import load_raw, load_split_indices
from src.metrics import at_capacity, at_review_count, at_threshold, fpr_threshold, ranking


AE_WEIGHTS = (0.05, 0.10, 0.20, 0.35, 0.50)
SCORE_LOCATIONS = {'GMM': ROOT / 'results/scores', 'Autoencoder': PHASE2}


def tail_surprise(sorted_legitimate, scores):
    """Negative log smoothed empirical upper-tail probability under legitimate activity."""
    count = len(sorted_legitimate)
    at_least_as_large = count - np.searchsorted(sorted_legitimate, scores, side='left')
    return -np.log((1 + at_least_as_large) / (count + 1))


def fuse(reference, scores, ae_weight):
    gmm = tail_surprise(reference['GMM'], scores['GMM'])
    ae = tail_surprise(reference['Autoencoder'], scores['Autoencoder'])
    return (1 - ae_weight) * gmm + ae_weight * ae


def select_hybrid(labels, scores):
    reference = {model: np.sort(score[labels == 0]) for model, score in scores.items()}
    trials = []
    for weight in AE_WEIGHTS:
        candidate = fuse(reference, scores, weight)
        trials.append({'ae_weight': weight, 'gmm_weight': 1 - weight,
                       'validation_ap': average_precision_score(labels, candidate)})
    winner = max(trials, key=lambda trial: (trial['validation_ap'], -trial['ae_weight']))
    return reference, winner['ae_weight'], pd.DataFrame(trials)


def paired_ap_difference(labels, hybrid, gmm, repetitions=300, seed=26):
    rng = np.random.default_rng(seed)
    legitimate = np.flatnonzero(labels == 0)
    fraud = np.flatnonzero(labels == 1)
    differences = []
    for _ in range(repetitions):
        sample = np.r_[rng.choice(legitimate, len(legitimate)), rng.choice(fraud, len(fraud))]
        differences.append(average_precision_score(labels[sample], hybrid[sample]) -
                           average_precision_score(labels[sample], gmm[sample]))
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {'replicates': repetitions, 'seed': seed, 'lower': float(lower), 'upper': float(upper)}


def load_scores(split, rows):
    return {model: read_frozen_scores(folder / f'{model.lower()}_{split}.npz', rows)
            for model, folder in SCORE_LOCATIONS.items()}


def evaluate(validation_labels, validation_scores, test_labels, test_scores):
    ranks, points, capacities = [], [], []
    for model, validation in validation_scores.items():
        cutoffs = {budget: fpr_threshold(validation_labels, validation, budget) for budget in BUDGETS}
        for split, labels, scores in (
            ('validation', validation_labels, validation), ('test', test_labels, test_scores[model]),
        ):
            tag = {'model': model, 'split': split}
            ranks.append({**tag, **ranking(labels, scores)})
            for budget, threshold in cutoffs.items():
                points.append({**tag, 'fpr_budget': budget, 'threshold_hex': threshold.hex(),
                               **at_threshold(labels, scores, threshold)})
            for fraction in CAPACITIES:
                capacities.append({**tag, 'policy': 'fraction', **at_capacity(labels, scores, fraction)})
            capacities.append({**tag, 'policy': 'top 100', **at_review_count(labels, scores, 100)})
    return pd.DataFrame(ranks), pd.DataFrame(points), pd.DataFrame(capacities)


def plot_test_curves(labels, scores, ranking_table, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout='constrained')
    for (name, score), color in zip(scores.items(), ('#0A6978', '#B65625', '#5545A0')):
        precision, recall, _ = precision_recall_curve(labels, score)
        ap = ranking_table.loc[ranking_table.model.eq(name) & ranking_table.split.eq('test'),
                               'average_precision'].iloc[0]
        ax.step(recall[::-1], precision[::-1], where='post', color=color, label=f'{name} ({ap:.3f} AP)')
    ax.axhline(labels.mean(), color='#69747A', linestyle=':', label='Fraud prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02),
           title='Historical test set (previously inspected)')
    ax.legend()
    fig.savefig(out / 'test_pr_hybrid.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/raw/creditcard.csv')
    parser.add_argument('--out', type=Path, default=PHASE2 / 'hybrid')
    args = parser.parse_args()
    parts = load_split_indices(load_raw(args.data), SPLIT_PATH, args.data, (0.6, 0.2, 0.2))

    validation_labels = parts['validation'].Class.to_numpy()
    validation_scores = load_scores('validation', parts['validation'].index)
    reference, ae_weight, trials = select_hybrid(validation_labels, validation_scores)
    validation_scores['Hybrid'] = fuse(reference, validation_scores, ae_weight)

    # The validation reference and weight are fixed before loading the historical test scores.
    test_labels = parts['test'].Class.to_numpy()
    test_scores = load_scores('test', parts['test'].index)
    test_scores['Hybrid'] = fuse(reference, test_scores, ae_weight)
    ranking_table, operating, capacities = evaluate(validation_labels, validation_scores, test_labels, test_scores)
    interval = paired_ap_difference(test_labels, test_scores['Hybrid'], test_scores['GMM'])

    args.out.mkdir(parents=True, exist_ok=True)
    for filename, table in (('validation_weights', trials), ('ranking', ranking_table),
                            ('operating_points', operating), ('capacity', capacities)):
        table.to_csv(args.out / f'{filename}.csv', index=False)
    plot_test_curves(test_labels, test_scores, ranking_table, args.out)
    inputs = [SPLIT_PATH, *(folder / f'{name.lower()}_{split}.npz'
                          for name, folder in SCORE_LOCATIONS.items() for split in ('validation', 'test'))]
    with np.load(SPLIT_PATH, allow_pickle=False) as split_metadata:
        dataset_hash = split_metadata['dataset_sha256'].item()
    metadata = {
        'ae_weight': ae_weight, 'gmm_weight': 1 - ae_weight,
        'weight_selection': 'Maximize mixed-validation AP, prefer lower AE weight on ties.',
        'standalone_policy': 'Evaluate the original GMM and autoencoder scores as component-removal baselines.',
        'calibration': 'Smoothed empirical upper tails from legitimate validation scores only.',
        'historical_test_previously_inspected': True,
        'historical_paired_ap_difference_ci': interval,
        'dataset_sha256': dataset_hash,
        'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(ranking_table.to_string(index=False))
    print(f'Validation-selected AE weight: {ae_weight:.2f}')
    print(f'Historical paired hybrid-minus-GMM AP interval: [{interval["lower"]:.4f}, {interval["upper"]:.4f}]')


if __name__ == '__main__':
    main()
