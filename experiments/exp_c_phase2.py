"""Phase-2 experiments: ablations, sample-size scaling, A1-violation robustness,
and an Experiment-B ablation of the invariance filter.

All CPU-bound, ~15 min total.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scin.scm import make_env_data, make_ood_task
from scin.models import (notears_fit, scin_fit, edges_from_single,
                         edges_from_intersection, edges_from_scin)
from scin.metrics import prf, shd, cohens_d

RESULTS = Path(__file__).resolve().parents[1] / "results"

# hyperparameters carried over from Experiment A validation
HP = dict(lambda1=0.1, lambda3=0.1, lambda_ev=1.0, tau=0.2, var_tol=0.01)


def standardize(Xs):
    Xp = np.vstack(Xs)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-12
    return [(X - mu) / sd for X in Xs], sd


def scin_pipeline(data, sd, seed, lam1=HP["lambda1"], lam3=HP["lambda3"],
                  lev=HP["lambda_ev"], use_inv_stat=True):
    """Warm-started SCIN fit + extraction. Returns binary adjacency."""
    Xs_std, sd_ = standardize(data["Xs"])
    Xp = np.vstack(Xs_std)
    W_pool = notears_fit(Xp, lambda1=lam1, lambda_ev=lev, scales=sd_, seed=seed)
    _, Ws = scin_fit(Xs_std, lambda1=lam1, lambda3=lam3, lambda_ev=lev,
                     scales=sd_, seed=seed, W_init=W_pool)
    if use_inv_stat:
        return edges_from_scin(Ws, HP["tau"], HP["var_tol"])
    st = np.stack(Ws)
    return (np.abs(st).mean(axis=0) > HP["tau"]).astype(float)


def ablation_a():
    """Component ablation on invariant-subgraph recovery (5 seeds)."""
    configs = [
        ("SCIN (full)", {}),
        ("SCIN w/o invariance stat", {"use_inv_stat": False}),
        ("SCIN w/o per-env deltas", {"lam3": 1e8}),   # pins deltas ~ 0
        ("SCIN w/o equal-variance crit.", {"lev": 0.0}),
    ]
    out = {}
    for name, kw in configs:
        rows = []
        for seed in range(5):
            data = make_env_data(seed=seed)
            _, sd = standardize(data["Xs"])
            est = scin_pipeline(data, sd, seed, **kw)
            p, r, f = prf(data["W_invariant"], est)
            rows.append({"precision": p, "recall": r, "f1": f,
                         "shd": shd(data["W_invariant"], est)})
        out[name] = {k: (float(np.mean([r[k] for r in rows])),
                         float(np.std([r[k] for r in rows]))) for k in rows[0]}
        print(f"  [ablation] {name}: F1 {out[name]['f1'][0]:.3f}±{out[name]['f1'][1]:.3f}")
    return out


def scaling_n():
    """Sample-size scaling, 3 seeds, SCIN vs NOTEARS-pooled."""
    out = {}
    for n in [250, 500, 1000, 2000]:
        rows_s, rows_n = [], []
        for seed in range(3):
            data = make_env_data(n_per_env=n, seed=seed)
            Xs_std, sd = standardize(data["Xs"])
            Xp = np.vstack(Xs_std)
            est_s = scin_pipeline(data, sd, seed)
            p, r, f = prf(data["W_invariant"], est_s)
            rows_s.append(f)
            W = notears_fit(Xp, lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                            scales=sd, seed=seed)
            _, _, f_n = prf(data["W_invariant"], edges_from_single(W, HP["tau"]))
            rows_n.append(f_n)
        out[str(n)] = {"scin_f1": (float(np.mean(rows_s)), float(np.std(rows_s))),
                       "notears_f1": (float(np.mean(rows_n)), float(np.std(rows_n)))}
        print(f"  [scaling] n={n}: SCIN {out[str(n)]['scin_f1'][0]:.3f}  NOTEARS {out[str(n)]['notears_f1'][0]:.3f}")
    return out


def robustness_a1():
    """A1 violation: heterogeneous noise variances (5 seeds)."""
    out = {}
    for jitter in [0.0, 0.5, 1.0]:
        rows_s, rows_n = [], []
        for seed in range(5):
            data = make_env_data(seed=seed, noise_jitter=jitter)
            Xs_std, sd = standardize(data["Xs"])
            Xp = np.vstack(Xs_std)
            est_s = scin_pipeline(data, sd, seed)
            rows_s.append(prf(data["W_invariant"], est_s)[2])
            W = notears_fit(Xp, lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                            scales=sd, seed=seed)
            rows_n.append(prf(data["W_invariant"],
                              edges_from_single(W, HP["tau"]))[2])
        out[str(jitter)] = {"scin_f1": (float(np.mean(rows_s)), float(np.std(rows_s))),
                            "notears_f1": (float(np.mean(rows_n)), float(np.std(rows_n)))}
        print(f"  [A1-violation jitter={jitter}]: SCIN {out[str(jitter)]['scin_f1'][0]:.3f}  NOTEARS {out[str(jitter)]['notears_f1'][0]:.3f}")
    return out


def ablation_b():
    """Exp B with/without the invariance filter (5 seeds)."""
    rows = []
    for seed in range(5):
        task = make_ood_task(n_per_env=2000, seed=seed)
        Xs_std, sd = standardize(task["Xs_train"])
        Xp = np.vstack(Xs_std)
        W_pool = notears_fit(Xp, lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                             scales=sd, seed=seed)
        _, Ws = scin_fit(Xs_std, lambda1=HP["lambda1"], lambda3=HP["lambda3"],
                         lambda_ev=HP["lambda_ev"], scales=sd, seed=seed,
                         W_init=W_pool)

        def select(use_inv):
            adj = edges_from_scin(Ws, HP["tau"], HP["var_tol"]) if use_inv \
                else (np.abs(np.stack(Ws).mean(0)) > HP["tau"]).astype(float)
            anc, frontier = set(), [6]
            while frontier:
                j = frontier.pop()
                for i in range(adj.shape[0]):
                    if adj[i, j] and i not in anc and i != 6:
                        anc.add(i)
                        frontier.append(i)
            sel = sorted(a for a in anc if a in task["feature_idx"]) \
                or list(task["feature_idx"])
            return sel

        for name, use_inv in [("SCIN (full)", True), ("SCIN w/o invariance", False)]:
            sel = select(use_inv)
            X = np.vstack([t[:, sel] for t in task["Xs_train"]])
            y = np.vstack([t[:, [task["y_idx"]]] for t in task["Xs_train"]]).ravel()
            Xb = np.hstack([X, np.ones((len(X), 1))])
            A = Xb.T @ Xb + 1e-2 * np.eye(Xb.shape[1]); A[-1, -1] -= 1e-2
            wb = np.linalg.solve(A, Xb.T @ y)
            w, b = wb[:-1], wb[-1]
            r = {"seed": seed, "method": name, "selected": sel}
            for tag, env in [("ood0", task["Xs_test"][0]), ("ood_neg", task["Xs_test"][1])]:
                r[f"mse_{tag}"] = float(np.mean((env[:, 6] - env[:, sel] @ w - b) ** 2))
            rows.append(r)
        print(f"  [exp B ablation] seed {seed}: full={rows[-2]['selected']} "
              f"noinv={rows[-1]['selected']}")
    out = {}
    for name in ["SCIN (full)", "SCIN w/o invariance"]:
        rs = [r for r in rows if r["method"] == name]
        out[name] = {k: (float(np.mean([r[k] for r in rs])),
                         float(np.std([r[k] for r in rs])))
                     for k in ["mse_ood0", "mse_ood_neg"]}
    return out, rows


def main():
    t0 = time.time()
    print("=== Ablation A (invariant-subgraph recovery) ===")
    abl_a = ablation_a()
    print("\n=== Sample-size scaling ===")
    scal = scaling_n()
    print("\n=== A1-violation robustness ===")
    rob = robustness_a1()
    print("\n=== Ablation B (Exp B, invariance filter) ===")
    abl_b, abl_b_rows = ablation_b()
    out = {"hp": HP, "ablation_a": abl_a, "scaling_n": scal,
           "a1_violation": rob, "ablation_b": abl_b, "ablation_b_rows": abl_b_rows,
           "runtime_s": time.time() - t0}
    (RESULTS / "phase2_ablations.json").write_text(json.dumps(out, indent=2))
    print(f"\nDone in {time.time()-t0:.0f}s -> results/phase2_ablations.json")


if __name__ == "__main__":
    main()
