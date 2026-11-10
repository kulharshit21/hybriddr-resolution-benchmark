import numpy as np
from sklearn.metrics import (cohen_kappa_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, average_precision_score)
import torch
from models import probabilities


def measure(labels, probs):
    labels, probs = np.asarray(labels, dtype=int), np.asarray(probs)
    pred = probs.argmax(1)
    qwk = cohen_kappa_score(labels, pred, labels=list(range(5)), weights='quadratic')
    confidence = probs.max(1)
    correct = pred == labels
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    referral = probs[:, 2:].sum(1)
    binary = labels >= 2
    cm = confusion_matrix(labels, pred, labels=list(range(5)))
    return {'n': len(labels), 'qwk': float(qwk) if np.isfinite(qwk) else None,
            'macro_f1': float(f1_score(labels, pred, labels=list(range(5)), average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
            'mae': float(np.abs(pred - labels).mean()), 'two_grade_error': float((np.abs(pred - labels) >= 2).mean()),
            'severe_undergrade': float(((labels >= 3) & (pred < 2)).sum() / max(1, (labels >= 3).sum())),
            'nll': float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12)).mean()),
            'brier': float(((probs - np.eye(5)[labels]) ** 2).sum(1).mean()), 'ece_15_equal_width': float(ece),
            'referral_auroc': float(roc_auc_score(binary, referral)) if len(np.unique(binary)) == 2 else None,
            'referral_auprc': float(average_precision_score(binary, referral)) if binary.any() else None,
            'confusion_matrix': cm.tolist(),
            'risk_coverage': [{'coverage': c, 'error_rate': float((~correct[np.argsort(-confidence)[:max(1, int(len(labels) * c))]]).mean())} for c in (.25, .5, .75, .9, 1.)]}


def fit_temperature(logits, labels, ordinal):
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    raw = torch.tensor(0., requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], max_iter=40, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        p = probabilities(logits, ordinal, raw.exp().clamp(.05, 20))
        loss = -p[torch.arange(len(labels)), labels].clamp_min(1e-12).log().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(raw.detach().exp().clamp(.05, 20))


def paired_bootstrap(frame_a, frame_b, repeats=2000, seed=2026):
    if frame_a.path.tolist() != frame_b.path.tolist() or frame_a.label.tolist() != frame_b.label.tolist():
        raise ValueError('Paired comparison needs identical ordered cohort')
    rng = np.random.default_rng(seed)
    labels = frame_a.label.to_numpy()
    a, b = frame_a.pred.to_numpy(), frame_b.pred.to_numpy()
    groups = [np.asarray(ix) for ix in frame_a.reset_index(drop=True).groupby('patient').indices.values()]
    values = []
    for _ in range(repeats):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        qa = cohen_kappa_score(labels[ix], a[ix], labels=list(range(5)), weights='quadratic')
        qb = cohen_kappa_score(labels[ix], b[ix], labels=list(range(5)), weights='quadratic')
        if np.isfinite(qa - qb):
            values.append(float(qa - qb))
    if not values:
        raise ValueError('No finite bootstrap replicates')
    return {'delta_qwk_ci95': np.quantile(values, [.025, .975]).tolist(),
            'finite_replicates': len(values), 'requested_replicates': repeats}
