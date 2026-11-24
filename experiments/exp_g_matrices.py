"""Experiment G-addendum — refit one Causal Chambers window and save the
fitted coefficient/adjacency matrices for Fig 7 (GT vs SCIN vs pooled
adjacency heatmaps).

Reuses the exact protocol of exp_g_chambers.py (seed 0, same HP, same
envs) and stores:
  - the 37-edge GT subgraph (binary),
  - NOTEARS-pooled binarized at tau=0.1,
  - SCIN binarized at tau=0.1, var_tol=0.02,
plus per-weight lists (i, j, weight) for both fits, so figures never
need refitting again.

Usage: python3 experiments/exp_g_matrices.py   (~3-4 min CPU)
Writes: results/exp_g_matrices.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.exp_g_chambers import (  # noqa: E402
    DROP_FIRST, ENV_NAMES, HP, LAM1_POOL, N_PER_ENV, RESULTS, VARS,
    ground_truth_subgraph, load_envs, load_zip)
from scin.models import edges_from_scin, edges_from_single, notears_fit, scin_fit  # noqa: E402

TAU = 0.1
SEED = 0


def wlist(W):
    W = np.asarray(W)
    return [{"i": int(i), "j": int(j), "w": round(float(W[i, j]), 4)}
            for i, j in zip(*np.nonzero(np.abs(W) > 0))]


def main():
    zf = load_zip()
    W_gt = ground_truth_subgraph()
    raw = []
    for en in ENV_NAMES:
        import pandas as pd
        with zf.open(
            f"lt_interventions_standard_v1/uniform_{en}.csv") as f:
            df = pd.read_csv(f)
        X = df[VARS].to_numpy(dtype=np.float64)[DROP_FIRST:]
        rng = np.random.default_rng(1000 * SEED + len(en))
        raw.append(X[rng.permutation(len(X))[:N_PER_ENV]])

    pooled = np.vstack(raw)
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
    envs = [(X - mu) / sd for X in raw]

    W_pool = notears_fit(np.vstack(envs), lambda1=LAM1_POOL,
                         lambda_ev=HP["lambda_ev"], scales=sd, seed=SEED)
    _, Ws = scin_fit(envs, lambda1=HP["lambda1"], lambda3=HP["lambda3"],
                     lambda_ev=HP["lambda_ev"], scales=sd, seed=SEED,
                     W_init=W_pool)

    adj_pool = edges_from_single(W_pool, TAU)
    adj_scin = edges_from_scin(Ws, TAU, HP["var_tol"])

    out = {
        "config": {"seed": SEED, "tau": TAU, "var_tol": HP["var_tol"],
                   "n_nodes": len(VARS), "vars": VARS, "envs": ENV_NAMES,
                   "n_gt_edges": int(W_gt.sum())},
        "W_gt": wlist(W_gt),
        "adj_gt": wlist(W_gt),
        "W_pool_list": wlist(W_pool),
        "adj_pool": wlist(adj_pool),
        "scin_mean_abs_list": wlist(np.mean(np.abs(Ws), axis=0)),
        "adj_scin": wlist(adj_scin),
    }
    (RESULTS / "exp_g_matrices.json").write_text(json.dumps(out, indent=2))

    n_gt, n_scin, n_pool = len(out["adj_gt"]), len(out["adj_scin"]), \
        len(out["adj_pool"])
    print(f"saved results/exp_g_matrices.json | GT {n_gt} edges | "
          f"SCIN {n_scin} | pooled {n_pool}")


if __name__ == "__main__":
    main()
