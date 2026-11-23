"""Experiment A: invariant-subgraph recovery on synthetic SCMs.

Methods: CorrThresh, NOTEARS-pooled, NOTEARS-per-env+intersection, SCIN (ours).
Hyperparameters selected on a held-out validation seed; reported on 5 test seeds.
Ground truth: invariant subgraph of the true DAG (edges whose mechanism is
identical in every environment).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scin.scm import make_env_data
from scin.models import (notears_fit, scin_fit, edges_from_single,
                         edges_from_intersection, edges_from_scin)
from scin.metrics import prf, shd

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

VAL_SEED = 100
TEST_SEEDS = [0, 1, 2, 3, 4]

D, N_EDGES, N_ENVS, N_PER_ENV, N_PERT = 10, 10, 2, 1000, 3

TAUS = [0.05, 0.1, 0.15, 0.2, 0.3]
LAM1S = [0.02, 0.05, 0.1]
LAM3S = [0.001, 0.01, 0.1]
EVLAMS = [0.0, 0.1, 1.0]
VAR_TOLS = [0.005, 0.01, 0.02]


def gen(seed):
    return make_env_data(d=D, n_edges=N_EDGES, n_envs=N_ENVS,
                         n_per_env=N_PER_ENV, n_perturbed=N_PERT, seed=seed)


def standardize(Xs):
    """Column-standardize with statistics pooled across environments (the same
    transform is applied to every environment, preserving coefficient
    invariance). Stabilizes the matrix-exponential acyclicity constraint.
    Returns (standardized envs, per-column stds)."""
    Xp = np.vstack(Xs)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-12
    return [(X - mu) / sd for X in Xs], sd


def corr_edges(Xs, tau):
    C = np.corrcoef(np.vstack(Xs), rowvar=False)
    return (np.abs(C) > tau).astype(float)


def evaluate(name, est_bin, data):
    p, r, f = prf(data["W_invariant"], est_bin)
    return {"method": name, "precision": p, "recall": r, "f1": f,
            "shd": shd(data["W_invariant"], est_bin)}


def main():
    t_start = time.time()

    # ---------------- hyperparameter selection on validation seed ------------
    print("=== Validation phase (seed", VAL_SEED, ") ===")
    val = gen(VAL_SEED)
    val["Xs"], sd = standardize(val["Xs"])
    Xp = np.vstack(val["Xs"])

    best = {}

    # CorrThresh
    scores = [(evaluate("corr", corr_edges(val["Xs"], t), val)["f1"], t) for t in TAUS]
    best["corr_tau"] = max(scores)[1]

    # NOTEARS pooled
    scores = []
    for l1 in LAM1S:
        for lev in EVLAMS:
            W = notears_fit(Xp, lambda1=l1, lambda_ev=lev, scales=sd, seed=VAL_SEED)
            for t in TAUS:
                scores.append((evaluate("notears", edges_from_single(W, t), val)["f1"], l1, lev, t))
    best["notears_lam1"], best["notears_lev"], best["notears_tau"] = max(scores)[1:]

    # NOTEARS per-env + intersection
    scores = []
    for l1 in LAM1S:
        for lev in EVLAMS:
            Ws = [notears_fit(Xe, lambda1=l1, lambda_ev=lev, scales=sd, seed=VAL_SEED)
                  for Xe in val["Xs"]]
            for t in TAUS:
                scores.append((evaluate("intx", edges_from_intersection(Ws, t), val)["f1"], l1, lev, t))
    best["intx_lam1"], best["intx_lev"], best["intx_tau"] = max(scores)[1:]

    # SCIN
    scores = []
    for l1 in LAM1S:
        for lev in EVLAMS:
            W_pool = notears_fit(Xp, lambda1=l1, lambda_ev=lev, scales=sd, seed=VAL_SEED)
            for l3 in LAM3S:
                _, Ws = scin_fit(val["Xs"], lambda1=l1, lambda3=l3, lambda_ev=lev,
                                 scales=sd, seed=VAL_SEED, W_init=W_pool)
                for t in TAUS:
                    for vt in VAR_TOLS:
                        b = edges_from_scin(Ws, t, vt)
                        scores.append((evaluate("scin", b, val)["f1"], l1, l3, lev, t, vt))
    (best["scin_lam1"], best["scin_lam3"], best["scin_lev"],
     best["scin_tau"], best["scin_vtol"]) = max(scores)[1:]

    print("selected hyperparameters:", json.dumps(best, indent=2))

    # ------------------------------ test seeds ------------------------------
    print("=== Test phase (seeds", TEST_SEEDS, ") ===")
    results = []
    for seed in TEST_SEEDS:
        data = gen(seed)
        data["Xs"], sd = standardize(data["Xs"])
        Xp = np.vstack(data["Xs"])

        results.append(evaluate("CorrThresh",
                                corr_edges(data["Xs"], best["corr_tau"]), data))

        W = notears_fit(Xp, lambda1=best["notears_lam1"],
                        lambda_ev=best["notears_lev"], scales=sd, seed=seed)
        results.append(evaluate("NOTEARS-pooled",
                                edges_from_single(W, best["notears_tau"]), data))

        Ws = [notears_fit(Xe, lambda1=best["intx_lam1"], lambda_ev=best["intx_lev"],
                          scales=sd, seed=seed) for Xe in data["Xs"]]
        results.append(evaluate("NOTEARS-env-intersection",
                                edges_from_intersection(Ws, best["intx_tau"]), data))

        W_pool = notears_fit(Xp, lambda1=best["scin_lam1"],
                             lambda_ev=best["scin_lev"], scales=sd, seed=seed)
        _, Ws_scin = scin_fit(data["Xs"], lambda1=best["scin_lam1"],
                              lambda3=best["scin_lam3"], lambda_ev=best["scin_lev"],
                              scales=sd, seed=seed, W_init=W_pool)
        results.append(evaluate("SCIN (ours)",
                                edges_from_scin(Ws_scin, best["scin_tau"],
                                                best["scin_vtol"]), data))
        print(f"seed {seed} done  [{time.time()-t_start:.0f}s elapsed]")

    # ------------------------------- aggregate ------------------------------
    summary = {}
    for m in ["CorrThresh", "NOTEARS-pooled", "NOTEARS-env-intersection", "SCIN (ours)"]:
        rows = [r for r in results if r["method"] == m]
        summary[m] = {k: (float(np.mean([r[k] for r in rows])),
                          float(np.std([r[k] for r in rows])))
                      for k in ["precision", "recall", "f1", "shd"]}

    out = {"hyperparameters": best, "val_seed": VAL_SEED,
           "test_seeds": TEST_SEEDS, "config": {
               "d": D, "n_edges": N_EDGES, "n_envs": N_ENVS,
               "n_per_env": N_PER_ENV, "n_perturbed": N_PERT},
           "per_seed": results, "summary": summary}
    (RESULTS / "exp_a_graph_recovery.json").write_text(json.dumps(out, indent=2))

    print("\n=== Invariant-subgraph recovery (mean ± std over", len(TEST_SEEDS), "seeds) ===")
    print(f"{'method':<28}{'precision':>14}{'recall':>14}{'F1':>14}{'SHD':>12}")
    for m, s in summary.items():
        print(f"{m:<28}{s['precision'][0]:>7.3f}±{s['precision'][1]:.3f}"
              f"{s['recall'][0]:>7.3f}±{s['recall'][1]:.3f}"
              f"{s['f1'][0]:>7.3f}±{s['f1'][1]:.3f}{s['shd'][0]:>7.1f}±{s['shd'][1]:.1f}")


if __name__ == "__main__":
    main()
