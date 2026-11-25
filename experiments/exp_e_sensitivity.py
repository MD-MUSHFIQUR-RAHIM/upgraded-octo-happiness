"""Experiment E-sens — selection-sensitivity analysis for Exp E.

Motivation: in exp_e_confounded, seed 4 dropped the TRUE parent x2
(cross-env coefficient var 0.0154 > var_tol 0.01) while still excluding the
spurious s (mean|w| 0.038 < tau), giving SCIN OOD_neg 1.29 vs Oracle 1.00.
The hard threshold var_tol=0.01 was tuned on Exp A data, so the question is
whether (var_tol, tau) is knife-edge or whether there is a plateau.

Design: sweep var_tol in {0.005, 0.01, 0.02, 0.03, 0.05, 0.08} x
tau in {0.02, 0.05, 0.1, 0.2, 0.3} over the 5 Exp E seeds. For each cell
report mean OOD MSEs (test gamma 0.60 and -1.00), s-inclusion count, and
x2-dropping count. Reference points: ERM (all 6 predictors) and Oracle
(x1, x2).

Expected robust cell: s-exclusion is easy (s is excluded by mean|w| where
its true coefficient is real but its invariance is violated by gamma_e*delta
in several seeds); x2 inclusion wants larger var_tol. The interesting risk
is large var_tol keeping a spurious-but-luckily-flat s edge.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from scin.models import scin_fit, notears_fit
from exp_e_confounded import (build_env, ancestors_of, ood_mses, NODES, Y,
                              HP, N_PER_ENV, GAMMAS_TR, GAMMAS_TE)

SEEDS = range(5)
VAR_TOLS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
TAUS = [0.02, 0.05, 0.1, 0.2, 0.3]


def run_seed_cache(seed):
    """Fit once per seed; sweep only the (cheap) selection rule."""
    rng = np.random.default_rng(seed)
    Xs = [build_env(g, N_PER_ENV, rng) for g in GAMMAS_TR]
    Tes = [build_env(g, N_PER_ENV, rng) for g in GAMMAS_TE]

    pooled = np.vstack(Xs)
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
    Xs_std = [(X - mu) / sd for X in Xs]
    Tes_std = [(X - mu) / sd for X in Tes]

    W_pool = notears_fit(np.vstack(Xs_std).astype(np.float64),
                         lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                         scales=sd, seed=seed)
    _, Ws_e = scin_fit(
        [Xe.astype(np.float64) for Xe in Xs_std],
        lambda1=HP["lambda1"], lambda3=HP["lambda3"],
        lambda_ev=HP["lambda_ev"], scales=sd, seed=seed, W_init=W_pool)
    st = np.stack(Ws_e)

    ref = {"erm_ood": ood_mses(Xs_std, Tes_std, list(range(6))),
           "oracle_ood": ood_mses(Xs_std, Tes_std, [1, 2])}
    return st, Xs_std, Tes_std, ref


def main():
    fits = {}
    rows = []
    for seed in SEEDS:
        if seed not in fits:
            fits[seed] = run_seed_cache(seed)
        st, Xs_std, Tes_std, ref = fits[seed]
        for vt in VAR_TOLS:
            for t in TAUS:
                adj = ((st.var(axis=0) < vt) &
                       (np.abs(st).mean(axis=0) > t)).astype(float)
                sel = ancestors_of(adj, Y)
                ood = ood_mses(Xs_std, Tes_std, sel) if sel else [np.nan, np.nan]
                rows.append({
                    "seed": seed, "var_tol": vt, "tau": t,
                    "selected": sel, "n_sel": len(sel),
                    "s_included": 3 in sel, "x2_dropped": 2 not in sel,
                    "ood0": ood[0], "ood_neg": ood[1],
                    "erm_ood": ref["erm_ood"], "oracle_ood": ref["oracle_ood"]})

    # aggregate over seeds
    agg = {}
    for vt in VAR_TOLS:
        for t in TAUS:
            cell = [r for r in rows if r["var_tol"] == vt and r["tau"] == t]
            agg[f"{vt}|{t}"] = {
                "ood_neg_mean": round(float(np.nanmean([r["ood_neg"]
                                                        for r in cell])), 3),
                "ood0_mean": round(float(np.nanmean([r["ood0"]
                                                     for r in cell])), 3),
                "s_included": sum(r["s_included"] for r in cell),
                "x2_dropped": sum(r["x2_dropped"] for r in cell),
                "n_sel_mean": round(float(np.mean([r["n_sel"]
                                                   for r in cell])), 1)}

    erm_neg = float(np.mean([r["erm_ood"][1] for r in rows[:5]]))
    ora_neg = float(np.mean([r["oracle_ood"][1] for r in rows[:5]]))

    out = {"config": {"seeds": list(SEEDS), "var_tols": VAR_TOLS,
                      "taus": TAUS, "n_per_env": N_PER_ENV,
                      "gammas_tr": GAMMAS_TR, "gammas_te": GAMMAS_TE},
           "reference": {"erm_ood_neg": round(erm_neg, 3),
                         "oracle_ood_neg": round(ora_neg, 3)},
           "grid_mean": agg, "rows": rows}

    import json
    name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "results", "exp_e_sensitivity.json")
    with open(name, "w") as f:
        json.dump(out, f, indent=2)

    print(f"reference: ERM ood_neg={erm_neg:.3f}  Oracle={ora_neg:.3f}")
    print("\nOOD_neg mean grid (rows=var_tol, cols=tau) "
          "[x2 dropped / s included counts out of 5]")
    print("var_tol\\tau " + " ".join(f"{t:>8}" for t in TAUS))
    for vt in VAR_TOLS:
        line = []
        for t in TAUS:
            c = agg[f"{vt}|{t}"]
            line.append(f"{c['ood_neg_mean']:>5.2f}({c['x2_dropped']}/"
                        f"{c['s_included']})")
        print(f"{vt:>10} " + " ".join(line))

    # robust plateau detection: cells where s never included AND x2 never
    # dropped AND ood_neg < ERM
    robust = [k for k, c in agg.items()
              if c["s_included"] == 0 and c["x2_dropped"] == 0
              and c["ood_neg_mean"] < erm_neg]
    print("\nrobust cells (s never in, x2 never dropped, beats ERM):",
          robust if robust else "NONE — hard thresholds lack a plateau")


if __name__ == "__main__":
    main()
