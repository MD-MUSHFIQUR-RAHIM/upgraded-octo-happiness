"""Experiment B: OOD generalization under spurious-correlation shift.

Setup: 7-node SCM, continuous target (linear-Gaussian, consistent with A1);
y depends causally on x1, x2 (invariant mechanisms); spurious feature s
depends on y with environment-dependent strength beta
(train: 2.0 / 1.0; validation: 1.5; test: 0.0 and -2.0).

Methods:
- ERM: ridge regression on all features (pooled training envs).
- V-REx (Krueger et al. 2021): variance-of-risks penalty across training
  envs; lambda selected on an intermediate-beta validation environment.
- Oracle: ridge on the true causal parents.
- SCIN (ours): multi-environment structure learning -> select ancestors of y
  in the invariant subgraph -> ridge on selected features.

Metric: MSE (lower is better).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scin.scm import make_ood_task
from scin.models import notears_fit, scin_fit, edges_from_scin
from scin.metrics import cohens_d

torch.set_default_dtype(torch.float64)
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

SEEDS = list(range(10))
N_PER_ENV = 2000
Y_IDX = 6
REX_LAMBDAS = [1.0, 10.0, 100.0, 1000.0]

# SCIN hyperparameters carried over from Experiment A validation (structure
# learning is unsupervised, so no label leakage). See exp_a_graph_recovery.py.
SCIN_HP = dict(lambda1=0.1, lambda3=0.1, lambda_ev=1.0, tau=0.2, var_tol=0.01)


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def ridge_fit(X, y, l2=1e-2):
    """Closed-form ridge with intercept."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + l2 * np.eye(Xb.shape[1])
    A[-1, -1] -= l2  # don't penalize intercept
    wb = np.linalg.solve(A, Xb.T @ y)
    return wb[:-1], wb[-1]


def pooled_stats(envs, feats):
    Xp = np.vstack([t[:, feats] for t in envs])
    return Xp.mean(0), Xp.std(0) + 1e-12


def erm_model(task):
    feats = task["feature_idx"]
    X = np.vstack([t[:, feats] for t in task["Xs_train"]])
    y = np.vstack([t[:, [task["y_idx"]]] for t in task["Xs_train"]]).ravel()
    w, b = ridge_fit(X, y)
    return lambda T: T[:, feats] @ w + b


def oracle_model(task):
    feats = task["causal_features"]
    X = np.vstack([t[:, feats] for t in task["Xs_train"]])
    y = np.vstack([t[:, [task["y_idx"]]] for t in task["Xs_train"]]).ravel()
    w, b = ridge_fit(X, y)
    return lambda T: T[:, feats] @ w + b


def vrex_model(task, lambda_rex, seed=0, steps=4000, lr=0.02):
    """V-REx with a linear predictor; features standardized with pooled
    training statistics (the same transform applied at evaluation)."""
    torch.manual_seed(seed)
    feats = task["feature_idx"]
    mu, sd = pooled_stats(task["Xs_train"], feats)

    def std(T):
        return (T[:, feats] - mu) / sd

    envs = [(torch.as_tensor(std(t)), torch.as_tensor(t[:, task["y_idx"]]))
            for t in task["Xs_train"]]
    w = torch.zeros(len(feats), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        risks = torch.stack([((Xe @ w + b - ye) ** 2).mean() for Xe, ye in envs])
        obj = risks.sum() + lambda_rex * risks.var(unbiased=False)
        obj.backward()
        opt.step()
    wn, bn = w.detach().numpy(), float(b.detach())
    return lambda T: std(T) @ wn + bn


def scin_model(task, seed):
    """Multi-env structure learning -> ancestors of y in invariant subgraph."""
    feats_all = list(range(task["Xs_train"][0].shape[1]))  # includes y
    Xs = task["Xs_train"]
    Xp = np.vstack(Xs)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-12
    Xs_std = [(X - mu) / sd for X in Xs]
    Xp_std = np.vstack(Xs_std)

    W_pool = notears_fit(Xp_std, lambda1=SCIN_HP["lambda1"],
                         lambda_ev=SCIN_HP["lambda_ev"], scales=sd, seed=seed)
    _, Ws = scin_fit(Xs_std, lambda1=SCIN_HP["lambda1"], lambda3=SCIN_HP["lambda3"],
                     lambda_ev=SCIN_HP["lambda_ev"], scales=sd, seed=seed,
                     W_init=W_pool)
    adj = edges_from_scin(Ws, SCIN_HP["tau"], SCIN_HP["var_tol"])

    # ancestors of y in the estimated invariant DAG
    d = adj.shape[0]
    ancestors = set()
    frontier = [Y_IDX]
    while frontier:
        j = frontier.pop()
        for i in range(d):
            if adj[i, j] and i not in ancestors and i != Y_IDX:
                ancestors.add(i)
                frontier.append(i)
    selected = sorted(a for a in ancestors if a in task["feature_idx"])
    if not selected:  # fallback: all features (reported honestly)
        selected = list(task["feature_idx"])
    X = np.vstack([t[:, selected] for t in task["Xs_train"]])
    y = np.vstack([t[:, [task["y_idx"]]] for t in task["Xs_train"]]).ravel()
    w, b = ridge_fit(X, y)
    return (lambda T, s=selected, w=w, b=b: T[:, s] @ w + b), selected


def select_rex_lambda(task, seed):
    best_lam, best_mse = None, np.inf
    for lam in REX_LAMBDAS:
        f = vrex_model(task, lam, seed=seed)
        m = mse(task["Xs_val"][0][:, task["y_idx"]], f(task["Xs_val"][0]))
        if m < best_mse:
            best_mse, best_lam = m, lam
    return best_lam


def main():
    t0 = time.time()
    rows = []
    selections = []
    rex_lambdas = []
    for seed in SEEDS:
        task = make_ood_task(n_per_env=N_PER_ENV, seed=seed)

        def eval_model(name, f, tag_extra=None):
            out = {"seed": seed, "method": name}
            for tag, env in [("id", task["Xs_train"][0]), ("ood0", task["Xs_test"][0]),
                             ("ood_neg", task["Xs_test"][1])]:
                out[f"mse_{tag}"] = mse(env[:, task["y_idx"]], f(env))
            rows.append(out)

        eval_model("ERM", erm_model(task))

        lam = select_rex_lambda(task, seed)
        rex_lambdas.append(lam)
        eval_model("V-REx", vrex_model(task, lam, seed=seed))

        eval_model("Oracle", oracle_model(task))

        f, sel = scin_model(task, seed)
        selections.append(sel)
        eval_model("SCIN (ours)", f)
        print(f"seed {seed}: SCIN selected {sel}, V-REx lambda={lam:g}  "
              f"[{time.time()-t0:.0f}s]")

    # aggregate (lower MSE is better)
    summary = {}
    methods = ["ERM", "V-REx", "Oracle", "SCIN (ours)"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        summary[m] = {k: (float(np.mean([r[k] for r in rs])),
                          float(np.std([r[k] for r in rs])))
                      for k in ["mse_id", "mse_ood0", "mse_ood_neg"]}

    # paired Cohen's d vs ERM on OOD MSE (positive = SCIN/V-REx better)
    stats = {}
    for m in ["V-REx", "SCIN (ours)"]:
        a = np.array([r["mse_ood0"] for r in rows if r["method"] == m])
        b_ = np.array([r["mse_ood0"] for r in rows if r["method"] == "ERM"])
        an = np.array([r["mse_ood_neg"] for r in rows if r["method"] == m])
        bn = np.array([r["mse_ood_neg"] for r in rows if r["method"] == "ERM"])
        stats[m] = {"cohens_d_vs_erm_ood0": cohens_d(b_, a),
                    "cohens_d_vs_erm_ood_neg": cohens_d(bn, an)}

    out = {"n_seeds": len(SEEDS), "n_per_env": N_PER_ENV, "scin_hp": SCIN_HP,
           "rex_lambdas": rex_lambdas, "scin_selections": selections,
           "per_seed": rows, "summary": summary, "effect_sizes": stats}
    (RESULTS / "exp_b_ood.json").write_text(json.dumps(out, indent=2))

    print("\n=== OOD generalization, MSE (mean ± std over", len(SEEDS), "seeds) ===")
    print(f"{'method':<14}{'ID':>16}{'OOD beta=0':>18}{'OOD beta=-2':>18}")
    for m in methods:
        s = summary[m]
        print(f"{m:<14}{s['mse_id'][0]:>8.4f}±{s['mse_id'][1]:.4f}"
              f"{s['mse_ood0'][0]:>9.4f}±{s['mse_ood0'][1]:.4f}"
              f"{s['mse_ood_neg'][0]:>9.4f}±{s['mse_ood_neg'][1]:.4f}")
    print("\nselection frequency per feature (x0,x1,x2,s,z1,z2):")
    for i, name in enumerate(["x0", "x1", "x2", "s", "z1", "z2"]):
        cnt = sum(1 for sel in selections if i in sel)
        print(f"  {name}: {cnt}/{len(selections)}")


if __name__ == "__main__":
    main()
