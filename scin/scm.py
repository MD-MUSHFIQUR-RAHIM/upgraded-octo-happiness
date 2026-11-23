"""Synthetic SCM data generation with environment-heterogeneous mechanisms.

Assumptions enforced here:
- A1: equal error variances (linear-Gaussian, all noise variances = 1)
  => the DAG is identifiable (Peters & Buhlmann 2014).
- A2: a chosen subset of edge mechanisms differs across environments.
"""

import numpy as np


def random_dag(d: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Erdos-Renyi-style DAG on d nodes with n_edges directed edges (upper-tri random perm)."""
    perm = rng.permutation(d)
    A = np.zeros((d, d), dtype=float)
    possible = [(i, j) for i in range(d) for j in range(i + 1, d)]
    idx = rng.choice(len(possible), size=n_edges, replace=False)
    for k in idx:
        i, j = possible[k]
        w = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)
        A[perm[i], perm[j]] = w  # parent perm[i] -> child perm[j]
    return A


def simulate_linear_sem(
    W: np.ndarray,
    n: int,
    rng: np.random.Generator,
    noise_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Sample n observations from X = X W + E (W[i,j] = i causes j). Row-major topological order."""
    d = W.shape[0]
    X = np.zeros((n, d))
    if noise_scale is None:
        noise_scale = np.ones(d)
    # topological order via repeated source removal
    Wt = W.copy()
    order = []
    remaining = set(range(d))
    while remaining:
        sources = [v for v in remaining if not any(Wt[u, v] != 0 for u in remaining)]
        if not sources:
            raise ValueError("W is not a DAG")
        order.extend(sorted(sources))
        remaining -= set(sources)
        for v in list(sources):
            Wt[v, :] = 0.0
    for j in order:
        parents = X @ W[:, j]
        X[:, j] = parents + rng.normal(0.0, noise_scale[j], size=n)
    return X


def make_env_data(
    d: int = 10,
    n_edges: int = 10,
    n_envs: int = 2,
    n_per_env: int = 1000,
    n_perturbed: int = 3,
    seed: int = 0,
):
    """Generate environments sharing a DAG, with `n_perturbed` edge mechanisms
    having env-2 coefficients different from env-1.

    Returns dict with:
      W_base        : (d,d) env-1 coefficient matrix (the true DAG, env 1)
      W_envs        : list of (d,d) per-env coefficient matrices
      W_invariant   : (d,d) binary adjacency of the invariant subgraph (ground truth for Exp A)
      W_full        : (d,d) binary adjacency of the full DAG
      Xs            : list of (n_per_env, d) samples per environment
    """
    rng = np.random.default_rng(seed)
    W_base = random_dag(d, n_edges, rng)
    edges = list(zip(*np.nonzero(W_base)))
    perturbed = set()
    if n_perturbed > 0 and len(edges) >= n_perturbed:
        perturbed = set(
            edges[k] for k in rng.choice(len(edges), size=n_perturbed, replace=False)
        )

    W_envs = []
    for e in range(n_envs):
        We = W_base.copy()
        if e > 0:
            for (i, j) in perturbed:
                # perturb mechanism strength (keep sign-flip possible for harder case)
                We[i, j] = W_base[i, j] * rng.choice([0.3, 1.8, -1.0])
        W_envs.append(We)

    Xs = [simulate_linear_sem(We, n_per_env, rng) for We in W_envs]

    W_invariant = np.zeros_like(W_base, dtype=float)
    for (i, j) in edges:
        coeffs = [We[i, j] for We in W_envs]
        if np.std(coeffs) < 1e-9:
            W_invariant[i, j] = W_base[i, j]
    return {
        "W_base": W_base,
        "W_envs": W_envs,
        "W_invariant": W_invariant,
        "W_full": (W_base != 0).astype(float),
        "Xs": Xs,
        "perturbed_edges": sorted(perturbed),
    }


def make_ood_task(n_per_env: int = 2000, seed: int = 0, betas=(2.0, 1.0),
                  test_betas=(0.0, -2.0), val_beta=1.5):
    """7-node SCM for Experiment B (continuous target, consistent with A1).

    Nodes: 0=x0, 1=x1, 2=x2, 3=s (spurious), 4=z1, 5=z2 (nuisance), 6=y (target)
    Structure (invariant): x0->x1, x0->x2, x1->y, x2->y
    Non-invariant:         y->s  with coefficient beta_e
    y = 1.5*x1 - 1.0*x2 + noise (continuous; linear-Gaussian, equal variances).

    NOTE: an earlier binary-target (thresholded y) variant FAILED for SCIN:
    thresholding breaks the linear-Gaussian assumption the structure learner
    relies on, and spurious edges into y survived ancestor selection
    (SCIN == ERM). Reported as a limitation; continuous target is the
    assumption-consistent setting.
    """
    d = 7
    rng = np.random.default_rng(seed)

    def sample_env(beta, n):
        X = np.zeros((n, d))
        X[:, 0] = rng.normal(size=n)                      # x0
        X[:, 1] = 1.0 * X[:, 0] + rng.normal(size=n)      # x1
        X[:, 2] = -1.0 * X[:, 0] + rng.normal(size=n)     # x2
        X[:, 4] = rng.normal(size=n)                      # z1 (nuisance)
        X[:, 5] = rng.normal(size=n)                      # z2 (nuisance)
        y = 1.5 * X[:, 1] - 1.0 * X[:, 2] + 1.0 * rng.normal(size=n)  # A1: unit noise
        X[:, 3] = beta * y + rng.normal(size=n)           # s spurious, env-dependent
        X[:, 6] = y
        return X

    Xs_train = [sample_env(b, n_per_env) for b in betas]
    Xs_test = [sample_env(b, n_per_env) for b in test_betas]
    Xs_val = [sample_env(val_beta, n_per_env)]
    feature_idx = [0, 1, 2, 3, 4, 5]  # everything except the target column
    y_idx = 6
    causal_features = [0, 1, 2]       # ancestors of y in the invariant subgraph
    return {
        "Xs_train": Xs_train,
        "Xs_test": Xs_test,
        "Xs_val": Xs_val,
        "feature_idx": feature_idx,
        "y_idx": y_idx,
        "causal_features": causal_features,
        "feature_names": ["x0", "x1", "x2", "s", "z1", "z2"],
    }
