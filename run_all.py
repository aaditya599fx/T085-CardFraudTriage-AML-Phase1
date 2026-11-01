#!/usr/bin/env python3
"""Run the integrated Phase 1 experiment after all role-owned modules are merged.

Use --check-only to inspect prerequisites without changing result files.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
COMPONENTS = {
    'data': ('src/data.py', 'Role 1 data/splits'),
    'features': ('src/features.py', 'Role 2 preprocessing'),
    'gmm_model': ('src/gmm_model.py', 'Role 3 GMM'),
    'baselines': ('src/baselines.py', 'Role 3 comparators'),
    'eda': ('src/eda.py', 'Role 1 EDA'),
    'metrics': ('src/metrics.py', 'Role 4 metrics'),
    'evaluation': ('src/evaluation.py', 'Role 4 evaluation'),
    'error_analysis': ('src/error_analysis.py', 'Role 5 error analysis'),
    'robustness': ('src/robustness.py', 'Role 5 robustness'),
}


def preflight(root, config_path, data_path, skip_robustness=False):
    """Read-only prerequisite checks. A quick run still requires evaluation/error analysis."""
    missing = []
    for module, (relative, owner) in COMPONENTS.items():
        if module == 'robustness' and skip_robustness:
            continue
        if not (root / relative).is_file():
            missing.append(f'{relative} ({owner})')
    for path, label in [(config_path, 'configuration'), (data_path, 'original dataset')]:
        if not path.is_file():
            missing.append(f'{path} ({label})')
    return missing


def validate_config(cfg, skip_robustness=False):
    required = ['seed', 'data_path', 'split_fractions', 'gmm', 'fpr_budgets',
                'review_fractions', 'bootstrap_repetitions']
    if not skip_robustness:
        required.append('robustness')
    absent = [key for key in required if key not in cfg]
    if absent:
        raise ValueError('Missing configuration keys: ' + ', '.join(absent))
    fractions = cfg['split_fractions']
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError('split_fractions must be three positive fractions summing to one.')
    if cfg['gmm'].get('selection') not in ['aic', 'bic']:
        raise ValueError('gmm.selection must be aic or bic.')
    if not cfg['gmm'].get('components') or not cfg['gmm'].get('covariance_types'):
        raise ValueError('The GMM component/covariance search must be nonempty.')
    if any(not 0 <= value < 1 for value in cfg['fpr_budgets']):
        raise ValueError('FPR budgets must be in [0, 1).')
    if any(not 0 < value <= 1 for value in cfg['review_fractions']):
        raise ValueError('Review fractions must be in (0, 1].')
    if any(not isinstance(k, int) or k < 1 for k in cfg.get('review_counts', [100])):
        raise ValueError('Review counts must be positive integers.')
    if cfg['bootstrap_repetitions'] < 1:
        raise ValueError('bootstrap_repetitions must be positive.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Override the original creditcard.csv path')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/config.yaml')
    parser.add_argument('--check-only', action='store_true', help='Check prerequisites without fitting or writing results')
    parser.add_argument('--skip-robustness', action='store_true', help='Quick run only; not a full submission reproduction')
    args = parser.parse_args(argv)
    import yaml
    if not args.config.is_file():
        parser.exit(2, f'Configuration missing: {args.config}\n')
    cfg = yaml.safe_load(args.config.read_text())
    try:
        validate_config(cfg, args.skip_robustness)
    except (ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f'Invalid configuration: {exc}\n')
    data_path = args.data.resolve() if args.data else ROOT / cfg['data_path']
    missing = preflight(ROOT, args.config, data_path, args.skip_robustness)
    if missing:
        parser.exit(2, 'Integration prerequisites are not yet present:\n- ' + '\n- '.join(missing) +
                    '\nMerge the listed role-owned modules and provide the original dataset before running.\n')
    if args.check_only:
        print('Preflight passed. No model fitting or result files were written.')
        return 0

    # Import the remaining role-owned interfaces only after the readable preflight.
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from threadpoolctl import threadpool_limits
    from src.data import load_raw, chronological_split, manifest, save_split_indices, load_split_indices
    from src.features import prepare, preprocess
    from src.gmm_model import fit_grid, score as gmm_score, plot_selection
    from src.baselines import fit_baselines, score as baseline_score
    from src.eda import plot_eda
    from src.evaluation import evaluate
    from src.error_analysis import analyze_errors
    if not args.skip_robustness:
        from src.robustness import run_robustness

    started = time.perf_counter()
    out = ROOT / 'results'
    for folder in ['tables', 'figures', 'scores', 'models']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    frame = load_raw(data_path)
    index_path = ROOT / 'data/processed/split_indices.npz'
    if index_path.exists():
        parts = load_split_indices(frame, index_path, data_path, cfg['split_fractions'])
    else:
        parts = chronological_split(frame, cfg['split_fractions'])
        save_split_indices(parts, index_path, data_path, cfg['split_fractions'])
    protocol = dict(
        status='configured', config=cfg,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        split_indices_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        selection='Training-only staged AIC/BIC; validation-only threshold calibration.',
        test_reuse='The test partition was inspected in an earlier experiment; this is follow-up evaluation.',
        robustness_split='Validation only; no test-based tuning.',
        robustness_requested=not args.skip_robustness,
    )
    (out / 'experiment_plan.json').write_text(json.dumps(protocol, indent=2) + '\n')
    # Remove a stale success marker before a real run; a failed run must not look complete.
    (out / 'run_metadata.json').unlink(missing_ok=True)
    details = manifest(parts, data_path)
    details['quality'] = dict(rows=len(frame), columns=len(frame.columns),
        missing_cells=int(frame.isna().sum().sum()), exact_duplicate_rows=int(frame.duplicated().sum()),
        duplicate_policy='Retained; tied timestamps stay in one partition.')
    (ROOT / 'data/processed/split_manifest.json').write_text(json.dumps(details, indent=2) + '\n')
    pd.DataFrame(details['splits']).to_csv(out / 'tables/split_summary.csv', index=False)
    train = parts['train']
    train.describe().to_csv(out / 'tables/training_descriptive_statistics.csv')
    plot_eda(train, out / 'figures')
    normal = train[train.Class == 0]
    xtrain, x, scaler = prepare(parts)
    with threadpool_limits(limits=2):
        gmm, grid = fit_grid(xtrain, cfg['gmm'], cfg['seed'])
        grid.to_csv(out / 'tables/gmm_grid.csv', index=False)
        plot_selection(grid, out / 'figures')
        models = {'gmm': gmm, **fit_baselines(xtrain, cfg['seed'])}
        scores = {}
        for name, model in models.items():
            scores[name] = {}
            for split in ['validation', 'test']:
                value = gmm_score(model, x[split]) if name == 'gmm' else baseline_score(name, model, x[split])
                if value.shape != (len(parts[split]),) or not np.isfinite(value).all():
                    raise RuntimeError(f'Invalid score contract: {name}/{split}')
                scores[name][split] = value
                np.savez_compressed(out / f'scores/{name}_{split}.npz',
                    score=value, source_row=parts[split].index.to_numpy())
            joblib.dump(model, out / f'models/{name}.joblib')
        joblib.dump(scaler, out / 'models/scaler.joblib')
        _, pipeline = preprocess(normal)
        joblib.dump(pipeline, out / 'models/preprocessing.joblib')
        result = evaluate(scores, parts, cfg, out)
        analyze_errors(parts, scores, out)
        if not args.skip_robustness:
            run_robustness(normal, parts['validation'], cfg['robustness'], out)
    metadata = dict(status='complete' if not args.skip_robustness else 'quick_run_complete',
        config=cfg, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, platform=platform.platform(), seconds=time.perf_counter() - started,
        train_legitimate_rows=len(xtrain), feature_count=xtrain.shape[1],
        protocol=protocol, robustness_completed=not args.skip_robustness)
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(result.to_string(index=False))
    print(f'Completed in {metadata["seconds"]:.1f} seconds.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
