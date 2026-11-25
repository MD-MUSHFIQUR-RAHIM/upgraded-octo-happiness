"""Experiment I — SCIN on the Sachs protein-signaling pathway (domain pilot).

Real-world multi-environment interventional data: 9 experimental conditions
(Sachs et al. 2005, Science), 11 phosphoproteins/phospholipids, literature
consensus ground truth (20 directed edges; the consensus graph contains
feedback loops, so acyclic estimators are scored both directed-F1 and on
the skeleton — we report both, and flag the DAG caveat).

Protocol mirrors the Causal Chambers analysis (Exp G):
- arcsinh(flow) transform (standard for cytometry), pooled standardization,
  equal-variance criterion in ORIGINAL units;
- per-seed random subsampling of n/env samples (chambers protocol n=500/env);
- DEFAULT hyperparameters transferred from synthetic (no retuning):
  lambda1=0.1, lambda3=0.1, lambda_ev=1.0, tau=0.1, var_tol=0.02;
- tau sweep reported, best-tau summary flagged as post-hoc;
- methods: NOTEARS-pooled, NOTEARS-per-env-intersection, SCIN (ours);
- 5 seeds.

Data: Zenodo record 7681811 (Ground_Truth.csv + condition CSVs); the 9 real
condition files are committed under data/sachs/ for reproducibility (the 5
icam2_* simulated files are excluded).

Writes results/exp_i_sachs_pathway.json.
"""
import json
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scin.models import (notears_fit, scin_fit, edges_from_single,
                         edges_from_intersection, edges_from_scin)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data", "sachs")
OUT = os.path.join(ROOT, "results", "exp_i_sachs_pathway.json")

SEEDS = list(range(5))
N_PER_ENV = 500
HP = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0,
      "tau": 0.1, "var_tol": 0.02}
TAU_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3]

CONDITIONS = ["cd3cd28", "cd3cd28_icam2", "cd3cd28_aktinhib", "cd3cd28_g0076",
              "cd3cd28_psitect", "cd3cd28_u0126", "cd3cd28_ly", "pma",
              "b2camp"]


def gt_matrix_(vars_):
    gt = pd.read_csv(os.path.join(DATA, "GroundTruth.csv"))
    ii = {n: i for i, n in enumerate(vars_)}
    W = np.zeros((len(vars_), len(vars_)))
    edges = []
    for _, r in gt.iterrows():
        if r["from"] in ii and r["to"] in ii:
            W[ii[r["from"]], ii[r["to"]]] = 1.0
            edges.append((r["from"], r["to"]))
    return W, edges


def prf_(Wt, We):
    t, e = Wt.astype(bool), We.astype(bool)
    tp = int(np.sum(t & e))
    fp = int(np.sum(~t & e))
    fn = int(np.sum(t & ~e))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def shd_(Wt, We):
    T, E = Wt.astype(bool), We.astype(bool)
    extra = int(np.sum(E & ~T))
    miss = 0
    rev = 0
    cnt = np.zeros_like(T)
    d = T.shape[0]
    for i in range(d):
        for j in range(d):
            if T[i, j] and not E[i, j]:
                if E[j, i] and not cnt[j, i]:
                    rev += 1
                    cnt[j, i] = True
                else:
                    miss += 1
    return int(extra + miss + rev)


def skeleton_(B):
    S = (np.abs(B) > 0).astype(float)
    return ((S + S.T) > 0).astype(float)


def load_envs_(n_per, seed):
    raw = []
    for k, cond in enumerate(CONDITIONS):
        df = pd.read_csv(os.path.join(DATA, f"{cond}.csv"))
        X = np.arcsinh(df.to_numpy(dtype=np.float64) / 5.0)   # cytometry arcsinh
        if seed is None:
            Xs = X
        else:
            rng = np.random.default_rng(100000 * seed + 97 * k)
            Xs = X[rng.permutation(len(X))[:n_per]]
        raw.append(Xs)
    pooled = np.vstack(raw)
    mu_, sd_ = pooled.mean(0), pooled.std(0) + 1e-12
    return [(X - mu_) / sd_ for X in raw], sd_


def main():
    vars_ = list(pd.read_csv(os.path.join(
        DATA, "cd3cd28.csv"), nrows=0).columns)
    W_gt, gt_edges = gt_matrix_(vars_)
    d = len(vars_)
    print(f"Sachs: d={d}, {len(CONDITIONS)} conditions, "
          f"{len(gt_edges)} GT edges ({int(W_gt.sum())} in matrix)")
    print(f"GT cycles present: "
          f"{int(np.sum((W_gt + W_gt.T) == 2)) // 2} reciprocal pairs")

    rows = []
    recovered = {}
    t0 = time.time()
    for seed in SEEDS:
        envs, sd = load_envs_(N_PER_ENV, seed)
        Xp = np.vstack(envs)
        W_p = notears_fit(Xp, lambda1=HP["lambda1"],
                          lambda_ev=HP["lambda_ev"], scales=sd, seed=seed)
        _, Ws = scin_fit(envs, lambda1=HP["lambda1"], lambda3=HP["lambda3"],
                         lambda_ev=HP["lambda_ev"], scales=sd, seed=seed,
                         W_init=W_p)
        st = np.stack(Ws)
        for tau in TAU_SWEEP:
            b_p = edges_from_single(W_p, tau)
            p, r, f = prf_(W_gt, b_p)
            rows.append({"seed": seed, "m": "NOTEARS-pooled", "tau": tau,
                         "precision": p, "recall": r, "f1": f,
                         "fdir": f, "shd": shd_(W_gt, b_p),
                         "skel_f1": prf_(skeleton_(W_gt),
                                         skeleton_(b_p))[2],
                         "n_edges": int(b_p.sum())})
            b_i = edges_from_intersection(Ws, tau)
            p, r, f = prf_(W_gt, b_i)
            rows.append({"seed": seed, "m": "NOTEARS-env-intersection",
                         "tau": tau, "precision": p, "recall": r, "f1": f,
                         "fdir": f, "shd": shd_(W_gt, b_i),
                         "skel_f1": prf_(skeleton_(W_gt),
                                         skeleton_(b_i))[2],
                         "n_edges": int(b_i.sum())})
            b_s = edges_from_scin(Ws, tau, HP["var_tol"])
            p, r, f = prf_(W_gt, b_s)
            rows.append({"seed": seed, "m": "SCIN (ours)", "tau": tau,
                         "precision": p, "recall": r, "f1": f,
                         "fdir": f, "shd": shd_(W_gt, b_s),
                         "skel_f1": prf_(skeleton_(W_gt),
                                         skeleton_(b_s))[2],
                         "n_edges": int(b_s.sum())})
        # default-HP SCIN selection at tau=0.1: which GT edges recovered?
        b_s = edges_from_scin(Ws, HP["tau"], HP["var_tol"])
        rec = [f"{vars_[i]}->{vars_[j]}"
               for i, j in zip(*np.nonzero(b_s)) if W_gt[i, j] > 0]
        miss = [f"{u}->{v}" for (u, v) in gt_edges
                if b_s[vars_.index(u), vars_.index(v)] == 0]
        recovered[seed] = {"recovered": rec, "missed": miss}
        print(f"[seed {seed}] done [{time.time()-t0:.0f}s] "
              f"default-HP SCIN recovered {len(rec)} GT edges")

    # summary: default tau + best tau per method
    summary = {}
    for m in sorted(set(r["m"] for r in rows)):
        r_def = [r for r in rows if r["m"] == m and r["tau"] == HP["tau"]]
        summary[m] = {
            "default_tau": {
                "f1": [round(float(np.mean([r["f1"] for r in r_def])), 3),
                       round(float(np.std([r["f1"] for r in r_def])), 3)],
                "precision": [round(float(np.mean([r["precision"]
                                                   for r in r_def])), 3),
                              round(float(np.std([r["precision"]
                                                  for r in r_def])), 3)],
                "recall": [round(float(np.mean([r["recall"]
                                                for r in r_def])), 3),
                           round(float(np.std([r["recall"]
                                               for r in r_def])), 3)],
                "shd": round(float(np.mean([r["shd"] for r in r_def])), 1),
                "skel_f1": [round(float(np.mean([r["skel_f1"]
                                                 for r in r_def])), 3),
                            round(float(np.std([r["skel_f1"]
                                                for r in r_def])), 3)],
                "n_edges": round(float(np.mean([r["n_edges"]
                                                for r in r_def])), 1),
            }}
        rr = [r for r in rows if r["m"] == m]
        best = max(rr, key=lambda r: r["f1"])
        r_best = [r for r in rr if r["tau"] == best["tau"]]
        summary[m]["best_tau"] = {
            "tau": best["tau"],
            "f1": [round(float(np.mean([r["f1"] for r in r_best])), 3),
                   round(float(np.std([r["f1"] for r in r_best])), 3)],
            "precision": [round(float(np.mean([r["precision"]
                                               for r in r_best])), 3),
                          round(float(np.std([r["precision"]
                                              for r in r_best])), 3)],
            "skel_f1": [round(float(np.mean([r["skel_f1"]
                                             for r in r_best])), 3),
                        round(float(np.std([r["skel_f1"]
                                            for r in r_best])), 3)],
            "shd": round(float(np.mean([r["shd"] for r in r_best])), 1),
            "n_edges": round(float(np.mean([r["n_edges"]
                                            for r in r_best])), 1)}
    sel_gt = {"fresh": gt_edges}
    out = {"config": {"d": d, "conditions": CONDITIONS,
                      "n_gt_edges": len(gt_edges), "n_envs": len(CONDITIONS),
                      "n_per_env": N_PER_ENV, "seeds": SEEDS,
                      "hp": HP, "tau_sweep": TAU_SWEEP,
                      "preprocessing": "arcsinh(./5), pooled standardization",
                      "gt_cyclic": True,
                      "note": "literature consensus GT contains feedback "
                              "loops -> acyclic estimators scored on the "
                              "skeleton in addition to directed F1"},
           "summary": summary, "recovered_at_default": recovered,
           "rows": rows, "time_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("\n=== summary (default tau) ===")
    for m, s in summary.items():
        print(f"{m:28s} f1={s['default_tau']['f1'][0]:.3f}"
              f"±{s['default_tau']['f1'][1]:.3f}"
              f" prec={s['default_tau']['precision'][0]:.3f}"
              f" skel_f1={s['default_tau']['skel_f1'][0]:.3f}"
              f" shd={s['default_tau']['shd']}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
