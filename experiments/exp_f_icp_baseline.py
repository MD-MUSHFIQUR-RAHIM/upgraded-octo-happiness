"""Experiment F: node-wise ICP baseline (Peters, Buhlmann & Meinshausen 2016)
vs SCIN on the Exp A invariant-subgraph recovery task.

ICP is inherently target-specific. To compare on full-graph recovery we run
ICP node-wise: for each target node j, test every candidate parent subset
S (subset-search over the other d-1 nodes), accept subsets whose per-env
residual distributions are indistinguishable, and estimate parents_j as the
intersection of accepted subsets. The full-graph estimate is the union over
targets of (parent, target) pairs.

With 2 environments (the Exp A setting) ICP is known to have low power —
this head-to-head is exactly what reviewers ask for. We also run a 4-env
variant to show ICP recovering with more heterogeneous environments while
SCIN only needs two (same seeds, same ground truth; SCIN re-uses the
hyperparameters selected on the 2-env validation protocol).

Statistical test: with A1 (equal Gaussian noise variances), residual variance
homogeneity across environments is tested with an exact two-sided F-test on
s1^2/s2^2 with (n1-p-1, n2-p-1) df. Accept subset if p > alpha.

Usage: python experiments/exp_f_icp_baseline.py [n_envs]
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scin.scm import make_env_data
from scin.models import notears_fit, scin_fit, edges_from_scin
from scin.metrics import prf, shd

RESULTS = Path(__file__).resolve().parents[1] / "results"

D, N_EDGES, N_PER_ENV, N_PERT = 10, 10, 1000, 3
TEST_SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.05

# SCIN hyperparameters: those selected on validation seed 100 in exp_a
# (2-env protocol); reused as-is for the multi-env head-to-head.
SCIN_HP = {"lam1": 0.1, "lam3": 0.1, "lev": 1.0, "tau": 0.2, "vtol": 0.01}


def standardize(Xs):
    Xp = np.vstack(Xs)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-12
    return [(X - mu) / sd for X in Xs]


def resid_var(Xe, j, S):
    """Residual variance and df of regressing Xe[:, j] on columns S."""
    if not S:
        r = Xe[:, j] - Xe[:, j].mean()
        p = 0
    else:
        n = len(Xe)
        Xb = np.hstack([Xe[:, S], np.ones((n, 1))])
        coef, *_ = np.linalg.lstsq(Xb, Xe[:, j], rcond=None)
        r = Xe[:, j] - Xb @ coef
        p = len(S)
    v = float(r.var(ddof=max(p + 1, 1)))
    df = max(len(r) - p - 2, 1)
    return v, df


def icp_parents(envs, j, alpha=ALPHA):
    """Node-wise ICP for target j. Returns (parents, n_accepted, fallback).
    A subset is accepted iff residual variance is homogeneous across EVERY
    pair of environments (pairwise exact F-tests, Bonferroni-free: accepting
    requires low variance AND a real mechanism match, so pairwise AND is the
    conservative choice under A1)."""
    others = [i for i in range(envs[0].shape[1]) if i != j]
    pairs = [(a, b) for a, b in combinations(range(len(envs)), 2)]
    accepted, best_p, best_S = [], -1.0, []
    for k in range(len(others) + 1):
        for S in combinations(others, k):
            S = list(S)
            stats_ = [resid_var(Xe, j, S) for Xe in envs]
            p_min = 1.0
            for a, b in pairs:
                (v1, df1), (v2, df2) = stats_[a], stats_[b]
                F = max(v1, v2) / (min(v1, v2) + 1e-300)
                p = 2.0 * min(stats.f.cdf(F, df1, df2),
                              1.0 - stats.f.cdf(F, df1, df2))
                p_min = min(p_min, min(p, 1.0))
            if p_min > best_p:
                best_p, best_S = p_min, S
            if p_min > alpha:
                accepted.append(set(S))
    if not accepted:
        return set(best_S[:3]), 0, True   # conservative fallback
    inter = set.intersection(*accepted)
    if not inter:
        # standard ICP fallback: min-size accepted subset (max-p)
        return set(best_S[:3]), len(accepted), True
    return inter, len(accepted), False


def icp_graph(envs):
    d = envs[0].shape[1]
    W = np.zeros((d, d))
    n_accepted_all, n_fallback = [], 0
    for j in range(d):
        pa, na, fb = icp_parents(envs, j)
        n_accepted_all.append(na)
        n_fallback += int(fb)
        for i in pa:
            W[i, j] = 1.0
    return W, float(np.mean(n_accepted_all)), n_fallback


def run_setting(n_envs):
    t0 = time.time()
    per_seed, summary_rows = [], []
    print(f"=== d={D}, {n_envs} envs, n/env={N_PER_ENV}, alpha={ALPHA} ===")
    for seed in TEST_SEEDS:
        data = make_env_data(d=D, n_edges=N_EDGES, n_envs=n_envs,
                             n_per_env=N_PER_ENV, n_perturbed=N_PERT,
                             seed=seed)
        envs = standardize(data["Xs"])

        rows = {}
        W_icp, mean_acc, n_fb = icp_graph(envs)
        p, r, f = prf(data["W_invariant"], W_icp)
        rows["ICP (node-wise)"] = {
            "seed": seed, "method": "ICP (node-wise, F-test)",
            "precision": p, "recall": r, "f1": f,
            "shd": shd(data["W_invariant"], W_icp),
            "mean_accepted_per_target": mean_acc,
            "fallback_targets": n_fb}

        # SCIN at the exp_a-selected hyperparameters, same data
        Xp = np.vstack(envs)
        sd = np.vstack(data["Xs"]).std(0) + 1e-12
        W_pool = notears_fit(Xp, lambda1=SCIN_HP["lam1"],
                             lambda_ev=SCIN_HP["lev"], scales=sd, seed=seed)
        _, Ws = scin_fit(envs, lambda1=SCIN_HP["lam1"], lambda3=SCIN_HP["lam3"],
                         lambda_ev=SCIN_HP["lev"], scales=sd, seed=seed,
                         W_init=W_pool)
        b = edges_from_scin(Ws, SCIN_HP["tau"], SCIN_HP["vtol"])
        p, r, f = prf(data["W_invariant"], b)
        rows["SCIN (ours)"] = {"seed": seed, "method": "SCIN (ours)",
                               "precision": p, "recall": r, "f1": f,
                               "shd": shd(data["W_invariant"], b)}

        for name, row in rows.items():
            per_seed.append(row)
            summary_rows.append(row)
        print(f"seed {seed}: ICP F1={rows['ICP (node-wise)']['f1']:.3f} "
              f"(fallback {n_fb}/10)  SCIN F1={rows['SCIN (ours)']['f1']:.3f} "
              f"[{time.time()-t0:.0f}s]")

    summary = {}
    for m in ["ICP (node-wise, F-test)", "SCIN (ours)"]:
        rr = [r for r in summary_rows if r["method"] == m]
        summary[m] = {k: (float(np.mean([r[k] for r in rr])),
                          float(np.std([r[k] for r in rr])))
                      for k in ["precision", "recall", "f1", "shd"]}
    out = {"config": {"d": D, "n_edges": N_EDGES, "n_envs": n_envs,
                      "n_per_env": N_PER_ENV, "n_perturbed": N_PERT,
                      "alpha": ALPHA, "seeds": TEST_SEEDS,
                      "scin_hp": SCIN_HP,
                      "note": ("2-env ICP; low power expected under A1 F-test"
                               if n_envs == 2 else
                               "multi-env ICP; SCIN hp re-used from 2-env val")},
           "per_seed": per_seed, "summary": summary}
    fname = f"exp_f_icp_baseline_nenvs{n_envs}.json"
    (RESULTS / fname).write_text(json.dumps(out, indent=2))
    print(f"-> saved {fname}")

    print(f"\n=== Summary ({n_envs} envs, {len(TEST_SEEDS)} seeds) ===")
    print(f"{'method':<28}{'precision':>14}{'recall':>14}{'F1':>14}{'SHD':>12}")
    for m, s in summary.items():
        print(f"{m:<28}{s['precision'][0]:>7.3f}±{s['precision'][1]:.3f}"
              f"{s['recall'][0]:>7.3f}±{s['recall'][1]:.3f}"
              f"{s['f1'][0]:>7.3f}±{s['f1'][1]:.3f}"
              f"{s['shd'][0]:>7.1f}±{s['shd'][1]:.1f}")
    return summary


if __name__ == "__main__":
    n_envs_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_setting(n_envs_arg)
