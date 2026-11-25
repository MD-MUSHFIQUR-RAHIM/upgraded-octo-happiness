"""Experiment H — controlled explanation of the chambers E-scaling decline.

Motivation (Exp G-max, results/exp_scinmax_gpu.json): on the Causal
Chambers, SCIN improved from E=4 to E=8 but DEGRADED at E=17 where three
strong interventions were added. Hypothesis: interventions far from the
reference level break the local-linearity of the observable mechanisms,
so the added environments carry model misspecification (noise in the
per-env linear fits), which inflates the cross-env coefficient variance
and makes the invariance gate reject TRUE edges.

Here we replay the same env ladder on synthetic linear data under three
regimes:

  A "clean":     every added environment is linear and clean; the ladder
                 should be monotonically IMPROVING (pure E effect).
  B "misspec":   environments beyond E=4 receive a mild monotone (tanh)
                 distortion (the same PNL transform as gpu_scaleup Part 2),
                 mimicking far-from Reference levels in a physical system;
                 prediction: non-monotonic / declining at large E.
  C "strong":    every added environment is clean but the mechanism *moves*
                 are much larger (harder heterogeneity, easy invariance);
                 prediction: monotonic improvement, faster than A — showing
                 movement STRENGTH itself is not the problem, misspecification
                 is.

If B declines where A/C rise, the chambers E=17 drop is explained: it is
intervention-strength-induced misspecification, not "too many
environments".

Setup: d=18 (chambers-sized), 22 edges, 5 moved mechanisms, n=500/env,
SEEDS = {0,1,2}, HP = chambers defaults (tau=0.1, var_tol=0.02, warm start).
Usage: python3 experiments/exp_h_env_ladder.py   (~45-60 min CPU)
Writes results/exp_h_env_ladder.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scin.models import edges_from_scin, notears_fit, scin_fit  # noqa: E402
from scin.scm import random_dag, simulate_linear_sem  # noqa: E402
from scin.metrics import prf, shd  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

D = 18
N_EDGES = 22
N_PERT = 5
N_PER = 500
EMON = 16                      # total env pool (1 reference + 15 interventions)
LADDER = [2, 4, 8, 16]
SEEDS = [0, 1, 2]
TAU = 0.1
VAR_TOL = 0.02
HP = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0}
DISTORT = 0.5                  # tanh distortion strength for regime B
STRONG_CHOICES = [0.1, 3.0, -2.5]
MILD_CHOICES = [0.3, 1.8, -1.0]


NOISE_MULT = 2.5               # residual-noise multiplier for regime 'noise'


def make_pool(seed, regime):
    """Return (W_invariant, list of E_max envs)."""
    rng = np.random.default_rng(seed)
    W_base = random_dag(D, N_EDGES, rng)
    edges = list(zip(*np.nonzero(W_base)))
    pert = [edges[k] for k in
            rng.choice(len(edges), size=N_PERT, replace=False)]
    choices = STRONG_CHOICES if regime == "strong" else MILD_CHOICES
    envs = []
    for e in range(EMON):
        We = W_base.copy()
        if e > 0:
            for (i, j) in pert:
                We[i, j] = W_base[i, j] * rng.choice(choices)
        noise_scale = None
        if regime == "noise" and e >= 4:
            # far-from-reference analogue: sensor-side (downstream) nodes get
            # noisier at extreme levels, so the added environments carry a
            # much larger residual variance while the structural signal is
            # unchanged. NOTE: a UNIFORM multiplier would be removed by
            # standardization (scale-invariance) — the partial inflation is
            # what breaks the fits.
            topo, remaining = [], set(range(D))
            Wt = We.copy()
            while remaining:
                src = [v for v in remaining
                       if not any(Wt[u, v] != 0 for u in remaining)]
                topo.extend(sorted(src)); remaining -= set(src)
                for v in src:
                    Wt[v, :] = 0.0
            noise_scale = np.ones(D)
            noise_scale[topo[len(topo) // 2:]] = NOISE_MULT
        X = simulate_linear_sem(We, N_PER, rng, noise_scale=noise_scale)
        if regime == "misspec" and e >= 4:
            # mild monotone (tanh) post-nonlinear distortion: the analogue of
            # a physical system operated far from its reference level
            X = X + DISTORT * 0.5 * np.tanh(X / 0.5)
        envs.append(X)
    W_inv = W_base.copy()
    for (i, j) in pert:
        W_inv[i, j] = 0.0
    return W_inv, envs


def main():
    t0 = time.time()
    regimes = os.environ.get(
        "REGIMES", "clean,misspec,strong,noise").split(",")
    out = {"config": {"d": D, "n_edges": N_EDGES, "n_perturbed": N_PERT,
                      "n_per_env": N_PER, "env_ladder": LADDER,
                      "seeds": SEEDS, "tau": TAU, "var_tol": VAR_TOL,
                      "hp": HP, "distort_strength": DISTORT,
                      "noise_mult": NOISE_MULT,
                      "regimes": {"clean": "all envs linear",
                                  "misspec": "envs >= 5 tanh-distorted",
                                  "strong": "clean, moved gaps 3-10x bigger",
                                  "noise": "envs >= 5 noise std x2.5"}},
           "rows": []}
    for regime in regimes:
        for seed in SEEDS:
            W_inv, pool = make_pool(seed, regime)
            for E_n in LADDER:
                envs = pool[:E_n]
                pooled = np.vstack(envs)
                mu, sd = pooled.mean(0), pooled.std(0) + 1e-12
                Xs = [(X - mu) / sd for X in envs]
                W_pool = notears_fit(np.vstack(Xs), lambda1=HP["lambda1"],
                                     lambda_ev=HP["lambda_ev"], scales=sd,
                                     seed=seed)
                _, Ws = scin_fit(Xs, lambda1=HP["lambda1"],
                                 lambda3=HP["lambda3"],
                                 lambda_ev=HP["lambda_ev"], scales=sd,
                                 seed=seed, W_init=W_pool)
                est = edges_from_scin(Ws, TAU, VAR_TOL)
                p, r, f = prf(W_inv, est)
                out["rows"].append(
                    {"regime": regime, "seed": seed, "E": E_n,
                     "precision": p, "recall": r, "f1": f,
                     "shd": shd(W_inv, est)})
                print(f"[{regime}] seed={seed} E={E_n}: F1={f:.3f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        # running regime summary
        rr = [x for x in out["rows"] if x["regime"] == regime]
        s = {}
        for E_n in LADDER:
            f1s = [x["f1"] for x in rr if x["E"] == E_n]
            s[E_n] = (round(float(np.mean(f1s)), 3),
                      round(float(np.std(f1s)), 3))
        out.setdefault("summary", {})[regime] = s
        print(f"== regime {regime} mean F1 by E: {s}", flush=True)
    (RESULTS / "exp_h_env_ladder.json").write_text(json.dumps(out, indent=1))
    print("\nsaved results/exp_h_env_ladder.json")
    for regime, s in out["summary"].items():
        print(f"  {regime:<9}: {s}")


if __name__ == "__main__":
    main()
