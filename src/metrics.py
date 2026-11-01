"""Validation-calibrated FPR policies and deterministic ranked review capacity."""
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

def check(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError('Labels and finite scores must be matching 1-D arrays.')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both legitimate and fraud observations are required.')
    return y, score

def ranking(y, score):
    y, score = check(y, score)
    p, r, _ = precision_recall_curve(y, score)
    return dict(average_precision=average_precision_score(y, score),
                pr_auc_trapezoidal=auc(r, p), roc_auc=roc_auc_score(y, score), base_rate=y.mean())

def fpr_threshold(y, score, budget):
    y, score = check(y, score)
    if not 0 <= budget < 1:
        raise ValueError('FPR budget must be in [0,1).')
    normal = np.sort(score[y == 0])[::-1]
    allowed = int(np.floor(budget * len(normal)))
    # Flag score > threshold. Strict inequality prevents ties exceeding the budget.
    return float(normal[allowed])

def at_threshold(y, score, threshold):
    y, score = check(y, score)
    flags = score > threshold
    tp = int(((y == 1) & flags).sum()); fp = int(((y == 0) & flags).sum())
    fn = int((y == 1).sum())-tp; tn = int((y == 0).sum())-fp
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
        recall=tp/(tp+fn), fpr=fp/(fp+tn), precision=tp/(tp+fp) if tp+fp else 0.,
        flagged=int(flags.sum()), review_fraction=float(flags.mean()))

def at_capacity(y, score, fraction):
    y, score = check(y, score)
    if not 0 < fraction <= 1:
        raise ValueError('Review fraction must be in (0,1].')
    k = max(1, int(np.floor(len(y) * fraction)))
    # Stable order breaks score ties using chronological row order, never labels.
    selected = np.argsort(-score, kind='stable')[:k]
    found = int(y[selected].sum())
    return dict(capacity_fraction=fraction, k=k, tp=found,
                precision_at_k=found/k, recall_at_k=found/int(y.sum()))

def bootstrap_ap(y, score, repetitions=500, seed=26):
    y, score = check(y, score)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    values = []
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(average_precision_score(y[idx], score[idx]))
    lo, hi = np.quantile(values, [.025, .975])
    return dict(ap_ci_lower=float(lo), ap_ci_upper=float(hi))


def at_review_count(y, score, k):
    y, score = check(y, score)
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(y):
        raise ValueError('Review count must be an integer between one and the batch size.')
    selected = np.argsort(-score, kind='stable')[:k]
    tp = int(y[selected].sum())
    return dict(k=int(k),tp=tp,precision_at_k=tp/k,recall_at_k=tp/int(y.sum()))


def evaluate_scores(y_true, anomaly_score, *, thresholds=None, review_fractions=(.005,), review_counts=(100,)):
    """Model-independent evaluation. Supplied thresholds must be frozen on validation.

    This function computes metrics only; it never fits a model or chooses a threshold.
    """
    y, s = check(y_true, anomaly_score)
    return dict(ranking=ranking(y,s),
        operating_points=[at_threshold(y,s,float(t)) for t in (thresholds if thresholds is not None else [])],
        review_fractions=[at_capacity(y,s,f) for f in review_fractions],
        review_counts=[at_review_count(y,s,int(k)) for k in review_counts])


def load_thresholds(path):
    """Reload frozen cutoffs exactly; default CSV decimal parsing can move tie boundaries."""
    import pandas as pd
    table = pd.read_csv(path, float_precision='round_trip')
    if 'threshold_hex' in table:
        table['threshold'] = table.threshold_hex.map(float.fromhex)
    if not np.isfinite(table.threshold.to_numpy()).all():
        raise ValueError('Frozen thresholds must be finite.')
    return table
