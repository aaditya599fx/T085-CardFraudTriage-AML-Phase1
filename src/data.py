"""Validated raw input and chronological partitions with indivisible time groups."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = [f'V{i}' for i in range(1, 29)] + ['Amount']
REQUIRED = ['Time'] + FEATURES + ['Class']

def load_raw(path):
    df = pd.read_csv(path)
    if set(df.columns) != set(REQUIRED):
        raise ValueError('Expected original ULB columns: Time, V1..V28, Amount, Class.')
    if df.empty or df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Empty data, missing values, or non-finite values require investigation.')
    if not set(df.Class.unique()).issubset({0, 1}) or df.Class.nunique() != 2:
        raise ValueError('Class must contain binary legitimate (0) and fraud (1) labels.')
    if (df.Amount < 0).any() or (df.Time < 0).any():
        raise ValueError('Time and Amount must be nonnegative.')
    # Keep source row identities for reproducible partition membership and capacity ties.
    df.index.name = 'source_row'
    return df.sort_values('Time', kind='stable')

def chronological_split(df, fractions=(.6, .2, .2)):
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError('Three positive split fractions summing to one are required.')
    n = len(df)
    times = df.Time.to_numpy()
    # Move boundaries right so equal timestamps never cross a partition.
    a = np.searchsorted(times, times[int(n * fractions[0])], side='right')
    b = np.searchsorted(times, times[int(n * sum(fractions[:2]))], side='right')
    if not 0 < a < b < n:
        raise ValueError('Insufficient distinct timestamps for three nonempty partitions.')
    parts = dict(train=df.iloc[:a].copy(), validation=df.iloc[a:b].copy(), test=df.iloc[b:].copy())
    if not parts['train'].Time.max() < parts['validation'].Time.min() < parts['test'].Time.min():
        raise AssertionError('Chronological ordering failed.')
    for name in ('validation', 'test'):
        if parts[name].Class.nunique() != 2:
            raise ValueError(f'{name} needs both classes for evaluation.')
    return parts

def manifest(parts, path):
    rows = []
    for name, frame in parts.items():
        rows.append(dict(split=name, rows=len(frame), fraud=int(frame.Class.sum()),
                         legitimate=int((frame.Class == 0).sum()), base_rate=float(frame.Class.mean()),
                         time_min=float(frame.Time.min()), time_max=float(frame.Time.max()),
                         index_sha256=hashlib.sha256(frame.index.to_numpy(dtype='int64').tobytes()).hexdigest()))
    return dict(dataset_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), splits=rows)


def save_split_indices(parts, path, dataset_path, fractions):
    """Persist membership with provenance; loading refuses another dataset or split policy."""
    arrays = {name: frame.index.to_numpy(dtype='int64') for name, frame in parts.items()}
    arrays['train_legitimate'] = parts['train'].index[parts['train'].Class == 0].to_numpy(dtype='int64')
    arrays['dataset_sha256'] = np.array(hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest())
    arrays['split_fractions'] = np.asarray(fractions, dtype=float)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_split_indices(df, path, dataset_path, fractions):
    with np.load(path, allow_pickle=False) as saved:
        checksum = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
        if saved['dataset_sha256'].item() != checksum:
            raise ValueError('Saved split belongs to another dataset; do not silently replace the frozen split.')
        if not np.allclose(saved['split_fractions'], fractions):
            raise ValueError('Split fractions differ from the frozen split.')
        arrays = {name:saved[name].copy() for name in ['train','validation','test','train_legitimate']}
    ids = np.concatenate([arrays[n] for n in ['train','validation','test']])
    if len(ids) != len(df) or len(np.unique(ids)) != len(df) or set(ids) != set(df.index):
        raise ValueError('Frozen split membership is overlapping, incomplete, or unknown.')
    expected = chronological_split(df, fractions)
    parts = {name:df.loc[arrays[name]].copy() for name in ['train','validation','test']}
    for name in parts:
        if not np.array_equal(parts[name].index, expected[name].index):
            raise ValueError('Frozen split ordering/boundaries disagree with the chronological policy.')
    legitimate = parts['train'].index[parts['train'].Class == 0].to_numpy(dtype='int64')
    if not np.array_equal(arrays['train_legitimate'], legitimate):
        raise ValueError('Frozen legitimate training membership is inconsistent.')
    return parts
