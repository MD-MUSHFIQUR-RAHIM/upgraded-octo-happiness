"""Experiment E — CONFOUNDED OOD task: spurious edge INTO the target.

Why: in Exp B the spurious variable s was a pure CHILD of y (s <- y), so the
invariance filter was not load-bearing: plain ancestor selection already
excluded s. Here the spurious variable sits INSIDE the target mechanism:

    x0 -> x1, x0 -> x2, x2 -> s,  y <- {x1, x2, s}
    y = 1.5*x1 - 1.0*x2 + gamma_e*s + eps_y        (gamma_e is env-specific)

Training envs use gamma in {0.40, 0.15}; test envs use OTHER gammas,
including a sign flip (-0.30). ERM fits one fixed coefficient for s, so it
cannot adapt and degrades. SCIN's per-env deltas make the s->y edge
coefficient vary across environments -> the invariance filter prunes it ->
s is not an invariant ancestor of y -> prediction recovers the oracle
model {x1, x2}.

This is the experiment that proves the invariance filter is load-bearing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)  # fp64 REQUIRED (see RESULTS.md)

from scin.models import scin_fit, notears_fit

SEEDS = range(5)
N_PER_ENV = 1500
NODES = ["x0", "x1", "x2", "s", "z1", "z2", "y"]
Y = 6
HP = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0,
      "tau": 0.1, "var_tol": 0.02}   # var_tol: see exp_e_sensitivity plateau
GAMMAS_TR = (0.80, 0.00)
GAMMAS_TE = (0.60, -1.00)
S_NOISE = 1.5


def build_env(gamma, n, rng):
    """Hand-built 7-node SEM, all noise unit-normal (A1 holds per env)."""
    X = np.zeros((n, 7))
    X[:, 0] = rng.normal(size=n)                   # x0
    X[:, 1] = 0.8 * X[:, 0] + rng.normal(size=n)   # x1
    X[:, 2] = -0.5 * X[:, 0] + rng.normal(size=n)  # x2
    X[:, 3] = 1.2 * X[:, 2] + S_NOISE * rng.normal(size=n)   # s (child of x2)
    X[:, 4] = rng.normal(size=n)                   # z1 (pure noise)
    X[:, 5] = rng.normal(size=n)                   # z2 (pure noise)
    X[:, 6] = (1.5 * X[:, 1] - 1.0 * X[:, 2] + gamma * X[:, 3]
               + rng.normal(size=n))               # y (confounded mechanism)
    return X


def ancestors_of(adj_bin, target):
    anc, frontier = set(), [target]
    adj = adj_bin.astype(bool)
    while frontier:
        j = frontier.pop()
        for i in range(adj.shape[0]):
            if adj[i, j] and i not in anc:
                anc.add(i)
                frontier.append(i)
    return sorted(anc)


def ood_mses(Xs_std, Te_stds, sel):
    """Closed-form ridge on selected columns (pooled train), OOD MSEs.

    NOTE: use vstack (NOT hstack!) for the y vector — hstack on (n,1)
    blocks interleaves the two environments' rows and silently destroys
    the X-y correspondence (found and fixed in this experiment)."""
    y_all = np.hstack([Xe[:, Y] for Xe in Xs_std])
    Xs_sel = np.vstack([Xe[:, sel] for Xe in Xs_std])
    Xb = np.hstack([Xs_sel, np.ones((Xs_sel.shape[0], 1))])
    A = Xb.T @ Xb + 1e-2 * np.eye(Xb.shape[1])
    A[-1, -1] -= 1e-2
    w = np.linalg.solve(A, Xb.T @ y_all)
    w_, b_ = w[:-1], w[-1]
    return [round(float(np.mean((Te[:, Y] - Te[:, sel] @ w_ - b_) ** 2)), 3)
            for Te in Te_stds]


def run_seed(seed):
    rng = np.random.default_rng(seed)
    Xs = [build_env(g, N_PER_ENV, rng) for g in GAMMAS_TR]
    Tes = [build_env(g, N_PER_ENV, rng) for g in GAMMAS_TE]

    pooled = np.vstack(Xs)
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
    Xs_std = [(X - mu) / sd for X in Xs]
    Tes_std = [(X - mu) / sd for X in Tes]

    res = {}

    # ---- ERM ridge: all 6 predictors (index 6 is y itself — no leakage!)
    res["erm_ood"] = ood_mses(Xs_std, Tes_std, list(range(6)))

    # ---- Oracle ridge: x1, x2 only
    res["oracle_ood"] = ood_mses(Xs_std, Tes_std, [1, 2])

    # ---- SCIN on the full 7-var stack (y is node 6)
    # CRITICAL: scales must be the RAW pooled stds (ev criterion is
    # evaluated in original units) — NOT the stds of standardized data.
    sd = pooled.std(0) + 1e-12
    W_pool = notears_fit(np.vstack(Xs_std).astype(np.float64),
                         lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                         scales=sd, seed=seed)
    _, Ws_e = scin_fit(
        [Xe.astype(np.float64) for Xe in Xs_std],
        lambda1=HP["lambda1"], lambda3=HP["lambda3"],
        lambda_ev=HP["lambda_ev"], scales=sd, seed=seed, W_init=W_pool)
    st = np.stack(Ws_e)
    adj_inv = ((st.var(axis=0) < HP["var_tol"]) &
               (np.abs(st).mean(axis=0) > HP["tau"])).astype(float)
    sel = ancestors_of(adj_inv, Y)
    res["scin_selected"] = sel
    res["scin_selected_names"] = [NODES[i] for i in sel]
    res["scin_ood"] = ood_mses(Xs_std, Tes_std, sel) if sel else [None, None]
    # diagnostic: cross-env variance of y's incoming coefficients
    var_y = st[:, :, Y].var(axis=0)
    res["var_y_parents"] = {NODES[i]: [round(float(var_y[i]), 4),
                                       round(float(np.abs(st[:, :, Y]).mean(axis=0)[i]), 3)]
                            for i in range(6)
                            if np.abs(st[:, :, Y]).mean(axis=0)[i] > 0.02}
    return res


def main():
    out = {"hp": HP, "n_per_env": N_PER_ENV,
           "gammas_tr": list(GAMMAS_TR), "gammas_te": list(GAMMAS_TE),
           "seeds": []}
    for seed in SEEDS:
        r = run_seed(seed)
        out["seeds"].append(r)
        print(f"seed {seed}: sel={r['scin_selected_names']} "
              f"ERM={r['erm_ood']} SCIN={r['scin_ood']} "
              f"Oracle={r['oracle_ood']}")
        if r["var_y_parents"]:
            print(f"          var(y-parents)={r['var_y_parents']}")

    agg = {k: np.array([s[k] for s in out["seeds"]], dtype=float)
           for k in ("erm_ood", "oracle_ood", "scin_ood")}
    out["mean"] = {k: [round(float(v), 3) for v in np.nanmean(v_, axis=0)]
                   for k, v_ in agg.items()}
    out["std"] = {k: [round(float(v), 3) for v in np.nanstd(v_, axis=0)]
                  for k, v_ in agg.items()}
    name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "results", "exp_e_confounded.json")
    with open(name, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== MEAN (test gamma=0.60 | test gamma=-0.30) ===")
    for k in out["mean"]:
        print(f"  {k}: {out['mean'][k]} +/- {out['std'][k]}")


if __name__ == "__main__":
    main()
