"""Analytic probability-convex oracle and label-free selector observables."""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score

FEATURES = ['S_confidence', 'F_confidence', 'S_margin', 'F_margin',
            'S_entropy', 'F_entropy', 'confidence_F_minus_S',
            'margin_F_minus_S', 'entropy_F_minus_S', 'prediction_disagreement',
            'probability_total_variation', 'Jensen_Shannon_divergence']


def observables(ps, pf):
    """No labels, receiver IDs, true-class margins, or TX-specific encodings."""
    ps, pf = np.asarray(ps, dtype=float), np.asarray(pf, dtype=float)
    ss, sf = np.sort(ps, axis=1), np.sort(pf, axis=1)
    cs, cf = ss[:, -1], sf[:, -1]
    ms, mf = cs - ss[:, -2], cf - sf[:, -2]
    entropy = lambda p: -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1) / np.log(p.shape[1])
    es, ef = entropy(ps), entropy(pf)
    avg = (ps + pf) / 2
    js = np.sum(ps * np.log(np.maximum(ps, 1e-300) / np.maximum(avg, 1e-300)), axis=1) / 2
    js += np.sum(pf * np.log(np.maximum(pf, 1e-300) / np.maximum(avg, 1e-300)), axis=1) / 2
    return np.column_stack([cs, cf, ms, mf, es, ef, cf-cs, mf-ms, ef-es,
                           ps.argmax(1) != pf.argmax(1), np.abs(ps-pf).sum(1)/2, js])


def convex_oracle(ps, pf, labels):
    """Exact scalar feasibility in float64, not a grid search.

argmax ties choose the smallest class index, as in numpy/torch. Thus the true
class must strictly beat lower-index classes and may tie higher-index ones.
Propagate open/closed interval bounds for every linear inequality.
"""
    ps, pf, y = np.asarray(ps, dtype=float), np.asarray(pf, dtype=float), np.asarray(labels, dtype=int)
    n, classes = ps.shape
    lower, upper = np.zeros(n), np.ones(n)
    lower_open, upper_open = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    valid = np.ones(n, dtype=bool)
    idx = np.arange(n)
    for j in range(classes):
        active = y != j
        strict = j < y
        a = ps[idx, y] - ps[:, j]
        b = (pf[idx, y] - pf[:, j]) - a
        zero = active & (b == 0)
        valid[zero & ((a < 0) | ((a == 0) & strict))] = False
        root = np.zeros(n)
        np.divide(-a, b, out=root, where=b != 0)
        pos, neg = active & (b > 0), active & (b < 0)
        replace = pos & (root > lower)
        same = pos & (root == lower)
        lower_open[replace] = strict[replace]
        lower_open[same] |= strict[same]
        lower[replace] = root[replace]
        replace = neg & (root < upper)
        same = neg & (root == upper)
        upper_open[replace] = strict[replace]
        upper_open[same] |= strict[same]
        upper[replace] = root[replace]
    feasible = valid & ((lower < upper) | ((lower == upper) & ~lower_open & ~upper_open))
    gamma = np.where(feasible, (lower+upper)/2, np.nan)
    return dict(feasible=feasible, lower=lower, upper=upper,
                lower_open=lower_open, upper_open=upper_open, witness=gamma)


def metric(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    cm = confusion_matrix(y, p, labels=np.arange(10))
    counts = cm.sum(1)
    recall = np.divide(np.diag(cm), counts, out=np.zeros(10, dtype=float), where=counts > 0)
    return dict(accuracy=float(np.mean(y == p)), macro_f1=float(f1_score(y, p, labels=np.arange(10), average='macro', zero_division=0)),
                recall=recall, support=counts, worst_TX_recall=float(recall.min()), worst_TX=int(recall.argmin()+31), confusion=cm)


def receiver_splits(heldout_receivers):
    """Leave one out only within the five RX unseen by the SAME CNN pair."""
    assert len(heldout_receivers) == 5 and len(set(heldout_receivers)) == 5
    return [(rx, [r for r in heldout_receivers if r != rx]) for rx in heldout_receivers]
