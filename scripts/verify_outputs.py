"""Verify a completed run's provenance, row alignment, policies and analysis outputs.

This checkpoint adds verification code; it does not assert that pending role modules
or their results have already been integrated into this repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Same external CSV override used by run_all.py')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/config.yaml')
    parser.add_argument('--skip-robustness', action='store_true', help='Verify quick-run outputs only; does not certify full reproduction')
    args = parser.parse_args(argv)
    prerequisite = [ROOT / 'src/metrics.py', ROOT / 'results/experiment_plan.json',
                    ROOT / 'results/run_metadata.json', args.config]
    missing = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
               for p in prerequisite if not p.is_file()]
    if missing:
        parser.exit(2, 'Verification prerequisites missing:\n- ' + '\n- '.join(missing) +
                    '\nMerge the remaining role modules and complete the pipeline first.\n')
    import numpy as np
    import pandas as pd
    import yaml
    from src.data import load_raw, load_split_indices
    from src.metrics import evaluate_scores, load_thresholds
    try:
        cfg = yaml.safe_load(args.config.read_text())
        data_path = args.data.resolve() if args.data else ROOT / cfg['data_path']
        tables = ROOT / 'results/tables'
        plan = json.loads((ROOT / 'results/experiment_plan.json').read_text())
        meta = json.loads((ROOT / 'results/run_metadata.json').read_text())
        require(meta['status'] in ['complete', 'quick_run_complete'], 'No successful current run marker.')
        require(plan['config_sha256'] == hashlib.sha256(args.config.read_bytes()).hexdigest(), 'Configuration changed after the run.')
        require(meta['config'] == cfg and plan['config'] == cfg, 'Configuration snapshots disagree.')
        require(plan['dataset_sha256'] == hashlib.sha256(data_path.read_bytes()).hexdigest(), 'Dataset changed after the run.')
        indices = ROOT / 'data/processed/split_indices.npz'
        require(plan['split_indices_sha256'] == hashlib.sha256(indices.read_bytes()).hexdigest(), 'Frozen split file changed after the run.')
        parts = load_split_indices(load_raw(data_path), indices, data_path, cfg['split_fractions'])
        require(meta['train_legitimate_rows'] == int((parts['train'].Class == 0).sum()), 'Training population count differs.')
        grid = pd.read_csv(tables / 'gmm_grid.csv', float_precision='round_trip')
        winner = grid[grid.selected]
        require(len(winner) == 1 and bool(winner.converged.iloc[0]), 'Exactly one converged selected GMM is required.')
        criterion = cfg['gmm']['selection']
        eligible = grid[grid.converged & np.isfinite(grid[criterion])]
        require(winner[criterion].iloc[0] == eligible[criterion].min(), 'Selected GMM violates the fixed training criterion.')
        expected_structural = len(cfg['gmm']['components']) * len(cfg['gmm']['covariance_types'])
        require(len(grid[grid.stage == 'structure']) == expected_structural, 'Structural grid coverage differs from configuration.')
        op = pd.read_csv(tables / 'operating_points.csv', float_precision='round_trip')
        ranking = pd.read_csv(tables / 'ranking_metrics.csv', float_precision='round_trip')
        capacity = pd.read_csv(tables / 'review_capacity.csv', float_precision='round_trip')
        cutoffs = load_thresholds(tables / 'thresholds_frozen_on_validation.csv')
        model_names = ['gmm', 'pca', 'isolation_forest', 'approx_ocsvm']
        require(len(op) == len(model_names) * 2 * len(cfg['fpr_budgets']), 'Missing or duplicated operating points.')
        for model in model_names:
            policies = cutoffs[cutoffs.model == model]
            require(np.array_equal(policies.fpr_budget.to_numpy(), cfg['fpr_budgets']), f'Policy budgets/order differ for {model}.')
            for split in ['validation', 'test']:
                frame = parts[split]
                with np.load(ROOT / f'results/scores/{model}_{split}.npz') as saved:
                    require(np.array_equal(saved['source_row'], frame.index), f'Wrong score row order: {model}/{split}')
                    scores = saved['score']
                measured = evaluate_scores(frame.Class.to_numpy(), scores,
                    thresholds=policies.threshold.to_numpy(), review_fractions=cfg['review_fractions'],
                    review_counts=cfg.get('review_counts', [100]))
                rows = ranking[(ranking.model == model) & (ranking.split == split)]
                require(len(rows) == 1, f'Ranking row count invalid: {model}/{split}')
                for name, value in measured['ranking'].items():
                    require(np.isclose(rows.iloc[0][name], value), f'Ranking mismatch: {model}/{split}/{name}')
                records = op[(op.model == model) & (op.split == split)]
                require(np.array_equal(records.fpr_budget.to_numpy(), cfg['fpr_budgets']), f'Operating-point order differs: {model}/{split}')
                for (_, record), actual in zip(records.iterrows(), measured['operating_points']):
                    for name in ['tp', 'fp', 'fn', 'tn', 'recall', 'fpr', 'precision', 'flagged']:
                        require(np.isclose(record[name], actual[name]), f'Confusion metric mismatch: {model}/{split}/{name}')
                    if split == 'validation':
                        require(record.fpr <= record.fpr_budget + 1e-12, f'Validation FPR budget exceeded: {model}')
                for policy, actuals in [('fraction', measured['review_fractions']), ('count', measured['review_counts'])]:
                    rows = capacity[(capacity.model == model) & (capacity.split == split) & (capacity.policy == policy)]
                    require(len(rows) == len(actuals), f'Review capacity coverage differs: {model}/{split}/{policy}')
                    for (_, record), actual in zip(rows.iterrows(), actuals):
                        for name in ['k', 'tp', 'precision_at_k', 'recall_at_k']:
                            require(np.isclose(record[name], actual[name]), f'Capacity metric mismatch: {model}/{split}/{name}')
        errors = pd.read_csv(tables / 'error_group_summary.csv')
        require(errors.rows.sum() == len(parts['test']), 'Error groups do not cover test observations.')
        default = op[(op.model == 'gmm') & (op.split == 'test') & np.isclose(op.fpr_budget, .005)]
        require(len(default) == 1, 'Error analysis requires a configured 0.5% validation FPR policy.')
        counts = errors.set_index('group').rows.to_dict()
        for group, metric in [('missed_fraud', 'fn'), ('false_positive', 'fp'), ('caught_fraud', 'tp'), ('true_negative', 'tn')]:
            require(counts.get(group, 0) == default.iloc[0][metric], f'Error-group mismatch: {group}')
        if not args.skip_robustness:
            require(meta['robustness_completed'] and meta['status'] == 'complete', 'This was only a quick run; full robustness verification is pending.')
            robust = pd.read_csv(tables / 'robustness_trials.csv')
            study = cfg['robustness']
            expected = (len(study['dimensions']) * len(study['covariance_types']) +
                        len(study['covariance_types']) * (len(study['seeds']) - 1) +
                        sum(value != 1 for value in study['train_fractions']))
            require(len(robust) == expected, 'Robustness trial count differs from the configured study.')
            require(set(robust.dimensions) == set(study['dimensions']), 'Robustness dimensions differ.')
            require(set(robust.seed) == set(study['seeds']), 'Robustness seeds differ.')
            require(set(robust.covariance) == set(study['covariance_types']), 'Robustness covariance types differ.')
            require(set(robust.train_fraction) == set(study['train_fractions']), 'Robustness sample fractions differ.')
        print('Verified provenance, training selection, eight score arrays, operating points, capacity policies and error groups.')
        print('Robustness: ' + ('not certified (quick-run verification).' if args.skip_robustness else 'full configured study verified.'))
        return 0
    except (ValueError, KeyError, OSError, IndexError) as exc:
        parser.exit(2, f'Output verification failed: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
