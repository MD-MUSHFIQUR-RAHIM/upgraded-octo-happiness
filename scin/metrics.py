"""Graph recovery and prediction metrics."""

import numpy as np


def binary_adj(W: np.ndarray, tau: float = 0.0) -> np.ndarray:
    return (np.abs(W) > tau).astype(int)


def prf(W_true_bin: np.ndarray, W_est_bin: np.ndarray):
    """Directed-edge precision / recall / F1."""
    true = W_true_bin.astype(bool)
    est = W_est_bin.astype(bool)
    tp = int(np.sum(true & est))
    fp = int(np.sum(~true & est))
    fn = int(np.sum(true & ~est))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def shd(W_true_bin: np.ndarray, W_est_bin: np.ndarray) -> int:
    """Structural Hamming distance for DAGs (missing + extra + reversed count 1 each)."""
    T = W_true_bin.astype(bool)
    E = W_est_bin.astype(bool)
    d = T.shape[0]
    extra = int(np.sum(E & ~T))
    missing_rev = 0
    missing = 0
    counted = np.zeros_like(T)
    for i in range(d):
        for j in range(d):
            if T[i, j] and not E[i, j]:
                if E[j, i] and not counted[j, i]:
                    missing_rev += 1  # reversed edge counts 1
                    counted[j, i] = True
                else:
                    missing += 1
    return extra + missing + missing_rev


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d."""
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")
