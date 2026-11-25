"""Experiment I-sweep — var_tol x tau sensitivity grid on Sachs.

Selection statistics are computed from the already-computed SCIN fits, so the
grid costs only refitting per seed once. Mirrors exp_e_sensitivity.py.
Writes results/exp_i_sachs_sweep.json.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scin.models import notears_fit, scin_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data", "sachs")
OUT = os.path.join(ROOT, "results", "exp_i_sachs_sweep.json")

CONDITIONS = ["cd3cd28", "cd3cd28_icam2", "cd3cd28_aktinhib", "cd3cd28_g0076",
              "cd3cd28_psitect", "cd3cd28_u0126", "cd3cd28_ly", "pma",
              "b2camp"]
SEEDS = list(range(5))
N_PER_ENV = 500
HP = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0}
VT_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
TAU_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


def prf_(Wt, We):
    t, e = Wt.astype(bool), We.astype(bool)
    tp = int(np.sum(t & e)); fp = int(np.sum(~t & e)); fn = int(np.sum(t & ~e))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def shd_(Wt, We):
    T, E = Wt.astype(bool), We.astype(bool)
    return int(np.sum(E & ~T) + np.sum(T & ~E & ~E.T))


def skeleton_(B):
    S = (np.abs(B) > 0).astype(float)
    return ((S + S.T) > 0).astype(float)


def main():
    vars_ = list(pd.read_csv(os.path.join(DATA, "cd3cd28.csv"),
                             nrows=0).columns)
    gt = pd.read_csv(os.path.join(DATA, "GroundTruth.csv"))
    ii = {n: i for i, n in enumerate(vars_)}
    W_gt = np.zeros((len(vars_),)) * 0
    W_gt = np.zeros((len(vars_), len(vars_)))
    for _, r in gt.iterrows():
        W_gt[ii[r["from"]], ii[r["to"]]] = 1.0

    def load_envs_(n_per, seed):
        raw = []
        for k, cond in enumerate(CONDITIONS):
            df = pd.read_csv(os.path.join(DATA, f"{cond}.csv"))
            X = np.arcsinh(df.to_numpy(dtype=np.float64) / 5.0)
            rng = np.random.default_rng(100000 * seed + 97 * k)
            raw.append(X[rng.permutation(len(X))[:n_per]])
        pooled = np.vstack(raw)
        mu_, sd_ = pooled.mean(0), pooled.std(0) + 1e-12
        return [(X - mu_) / sd_ for X in raw], sd_

    st_fits, pooled_fits, scale_fits = {}, {}, {}
    t0 = time.time()
    for seed in SEEDS:
        envs, sd = load_envs_(N_PER_ENV, seed)
        Xp = np.vstack(envs)
        W_p = notears_fit(Xp, lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                          scales=sd, seed=seed)
        _, Ws = scin_fit(envs, lambda1=HP["lambda1"], lambda3=HP["lambda3"],
                         lambda_ev=HP["lambda_ev"], scales=sd, seed=seed,
                         W_init=W_p)
        st_fits[seed] = np.stack(Ws)
        pooled_fits[seed] = W_p
        scale_fits[seed] = envs
        print(f"[seed {seed}] fits done [{time.time()-t0:.0f}s]")

    # pooled baseline per tau (per-seed, then averaged)
    rows = []
    for tau in TAU_GRID:
        ps_, rs_, fs_, ss_, sh_ = [], [], [], [], []
        for seed in SEEDS:
            b = (np.abs(pooled_fits[seed]) > tau).astype(float)
            p, r, f = prf_(W_gt, b)
            ps_.append(p); rs_.append(r); fs_.append(f)
            ss_.append(prf_(skeleton_(W_gt), skeleton_(b))[2])
            sh_.append(shd_(W_gt, b))
        rows.append({"m": "NOTEARS-pooled", "var_tol": None, "tau": tau,
                     "precision": round(float(np.mean(ps_)), 3),
                     "recall": round(float(np.mean(rs_)), 3),
                     "f1": round(float(np.mean(fs_)), 3),
                     "s_f1": round(float(np.mean(ss_)), 3),
                     "shd": round(float(np.mean(sh_)), 1),
                     "n_edges": round(float(sum(
                         (np.abs(pooled_fits[s]) > tau).sum()
                         for s in SEEDS)) / len(SEEDS), 1)
                     })

    grid_mean = {}
    for vt in VT_GRID:
        for tau in TAU_GRID:
            f1s, sf1s = [], []
            n_sel = 0
            for seed in SEEDS:
                st = st_fits[seed]
                b = ((st.var(0) < vt) & (np.abs(st).mean(0) > tau)).astype(float)
                f1s.append(prf_(W_gt, b)[2])
                sf1s.append(prf_(skeleton_(W_gt), skeleton_(b))[2])
                n_sel += b.sum()
            grid_mean[f"{vt}|{tau}"] = {
                "f1": round(float(np.mean(f1s)), 3),
                "f1_std": round(float(np.std(f1s)), 3),
                "skel_f1": round(float(np.mean(sf1s)), 3),
                "n_edges": round(float(n_sel) / len(SEEDS), 1)}
            rows.append({"m": "SCIN (ours)", "var_tol": vt, "tau": tau,
                         "precision": None, "recall": None,
                         "f1": round(float(np.mean(f1s)), 3),
                         "s_f1": round(float(np.mean(sf1s)), 3),
                         "shd": None,
                         "n_edges": round(float(n_sel) / len(SEEDS), 1)})
    best = max(grid_mean.items(), key=lambda kv: kv[1]["f1"])
    best_s = max(grid_mean.items(), key=lambda kv: kv[1]["skel_f1"])
    out = {"grid_mean": grid_mean, "rows": rows,
           "best_f1": {"key": best[0], **best[1]},
           "best_skel_f1": {"key": best_s[0], **best_s[1]},
           "pooled_best_f1": max(r["f1"] for r in rows
                                 if r["m"] == "NOTEARS-pooled"),
           "config": {"seeds": SEEDS, "n_per_env": N_PER_ENV,
                      "vt_grid": VT_GRID, "tau_grid": TAU_GRID, "hp": HP},
           "time_s": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("\nbest SCIN f1:", best[0], best[1])
    print("best SCIN skel_f1:", best_s[0], best_s[1])
    print("pooled best f1:", out["pooled_best_f1"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
