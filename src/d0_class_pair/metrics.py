"""Frozen, receiver-balanced class-pair diagnostic definitions."""
import numpy as np
from scipy.stats import spearmanr


def pair_metrics(logits, labels):
    c = logits.shape[1]
    predicted = logits.argmax(axis=1)
    counts = np.zeros((c, c), dtype=int)
    np.add.at(counts, (labels, predicted), 1)
    risk = counts / counts.sum(axis=1, keepdims=True)
    scale = np.sqrt(np.mean((logits - logits.mean(axis=1, keepdims=True)) ** 2, axis=1))
    # A constant-logit sample has zero difference for every pair.
    scale = np.where(scale == 0, 1.0, scale)
    records = []
    for i in range(c):
        mask = labels == i
        for j in range(c):
            if i == j:
                continue
            margin = logits[mask, i] - logits[mask, j]
            normalized = margin / scale[mask]
            row = dict(i=i, j=j, tx_i=i+31, tx_j=j+31,
                       confusion_count=int(counts[i, j]), confusion_risk=float(risk[i, j]),
                       pair_order_violation=float(np.mean(margin <= 0)))
            for name, values in [('raw', margin), ('normalized', normalized)]:
                row.update({name+'_margin_mean': float(np.mean(values)),
                            name+'_margin_median': float(np.median(values)),
                            name+'_margin_q05': float(np.quantile(values, .05))})
            records.append(row)
    return counts, risk, records


def offdiag(matrix):
    return matrix[~np.eye(len(matrix), dtype=bool)]


def predictors(reference_risks):
    """Only reference RX data enter this function, never probe labels/scores."""
    mean = np.mean(reference_risks, axis=0).copy()
    np.fill_diagonal(mean, 0)
    c = len(mean)
    errors = mean.sum(axis=1)
    dest = mean.sum(axis=0)
    class_only = np.repeat((errors / (c-1))[:, None], c, axis=1)
    factorized = np.zeros_like(mean)
    for i in range(c):
        weights = dest.copy()
        weights[i] = 0
        if weights.sum() == 0:
            weights[:] = 1
            weights[i] = 0
        factorized[i] = errors[i] * weights / weights.sum()
    return dict(pairwise_mean=offdiag(mean), source_class_only=offdiag(class_only),
                source_class_and_destination_factorized=offdiag(factorized),
                uniform=np.ones(c*(c-1)))


def topk_weights(scores, k=10):
    """Fractionally split boundary ties to select exactly k edges."""
    scores = np.asarray(scores)
    cutoff = np.sort(scores)[-k]
    above, equal = scores > cutoff, scores == cutoff
    weights = above.astype(float)
    weights[equal] = (k - above.sum()) / equal.sum()
    return weights


def evaluate_profile(scores, observed, k=10):
    rho = None if np.ptp(scores) == 0 or np.ptp(observed) == 0 else float(spearmanr(scores, observed)[0])
    total = observed.sum()
    capture = None if total == 0 else float(topk_weights(scores, k) @ observed / total)
    return rho, capture
