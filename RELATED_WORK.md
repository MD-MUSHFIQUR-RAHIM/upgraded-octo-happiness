# Related Work & Gap Analysis — SCIN (draft v1.0, 2026-11-24)

Purpose: position the paper honestly against the closest lines of work, and
document exactly what is new and what is not. Every entry below is load-bearing
for the paper's Related Work section.

## 1. Closest lines of work

### 1.1 NOTEARS-style continuous DAG learning
- **Zheng et al. 2018** — nonconvex acyclicity `tr(expm(W∘W)) − d = 0` turns
  structure learning into continuous optimization. Setting: pooled, i.i.d. data.
- **Pamfil et al. 2020, Yu et al. 2019 (DAG-GNN)** — deeper models, same pooled-data setting.
- **Gap we fill:** these methods pool all environments and therefore inherit
  the *pooled non-identifiability* problem — an edge that varies across
  environments can survive in the pooled DAG (T2 counterexample). SCIN learns
  **per-environment graphs W_e = W + Δ_e** around one shared skeleton and
  selects the edge set that is *mechanism-invariant* across environments.
  Empirical proof: Exp A F1 0.945 (SCIN) vs 0.851 (NOTEARS-pooled) on an
  invariant-subgraph task; Phase-2 pooled baseline drops further under shift.

### 1.2 Invariant Causal Prediction (ICP) and IRM family
- **Peters, Janzing, Schölkopf 2016 (ICP)** — subscript: enumerate subsets S,
  find those with invariant residual distribution across environments.
  Exponential in d; linear-regression assumption; linear Gaussian setting.
- **Arjovsky et al. 2019 (IRM)** — gradient-penalty invariant predictor.
  Known to fail to recover causal predictors even in simple settings
  (Rosenfeld et al. 2021: linear IRM needs pathologically many environments;
  Kamath et al. 2020: node-wise IRMs divergence results). Our own experience
  matches: on our tasks IRM-style penalties were unstable → we use **V-REx
  (Krueger et al. 2021)** as the invariance baseline; it barely improves on
  ERM in Exp B.
- **Gap we fill:** SCIN replaces ICP's exponential subset-search with a
  differentiable joint search: one pass fits both the graph structure and the
  invariance selection statistic in O(d²) parameters rather than O(2^d)
  subsets. The equal-variance criterion (Peters & Bühlmann 2014) gives
  orientation identifiability that IRM never exploits.

### 1.3 ICP extensions with structure learning
- **Invariance-based/structure-joint methods** — e.g. Strobl et al. (ICS), **Kamkar et al. (ICP-SS)**; recent principled approaches like
  ICP with sparsity boosting, and RECI/DCI-type heterogeneity methods;
  non-convex multi-env DAG learning such as DAG-ICP/HEJ (heterogeneous ERM), and
  Ghassami et al. 2018 (multi-env DAG learning from shifted distributions) —
  closest theory relative: identifiability from mechanism shifts; but these are combinatorial/optimization-based and scale poorly.
- **Gap we fill:** a *cheap, differentiable, GPU-friendly* version of the same
  idea with an explicit selection-time invariance test (Var_e(w_ij) + mean|w|
  thresholds), validated up to d = 200 at n = 2000/env in ≈ 24 s on a single
  GPU (~flat runtime in d).

### 1.4 OOD generalization benchmarks
- **DomainBed (Gulrajani & Lopez-Paz 2021)** — achievable-truth: invariance
  methods ≈ ERM on vision benchmarks. Consistent with our neural SCIN
  negative result; strengthens the paper's honesty: the *linear mechanistic*
  setting is where invariance into structure demonstrably pays, when (A1–A3)
  hold.
- **Gap we fill:** the paper explicitly separates *mechanism-level* shift
  (interventions on the causal mechanism — our Exp E sign-flip) from
  *spurious-encoder* shift, showing where the boundary of current invariance
  learning lies.

### 1.5 Equal-variance criterion
- **Peters & Bühlmann 2014** — identifiability of linear Gaussian DAGs under
  equal noise variances. We import this as assumption (A1) and as an
  *estimator component* (λ_ev penalty); Phase-2 ablations show this is
  the single most load-bearing piece (F1 0.945 → 0.244 without it).
  Credit is given; novelty is in composing it with multi-env deltas +
  invariance selection, not in the criterion itself.

## 2. Gap table (paper Table 1 candidate)

| Capability | NOTEARS/DAG-GNN | ICP | IRM / V-REx | Multi-env DAG (Ghassami+) | **SCIN (ours)** |
|---|---|---|---|---|---|
| Continuous, differentiable structure learning | ✓ | – | – | ~ | ✓ |
| Per-environment mechanism shift handling | – | ✓ | ✓ | ✓ | ✓ |
| Joint structure + invariance in one optimization | – | – | ~ | – | ✓ |
| Identifiable orientation (equal-variance) | – | ✓ | – | – | ✓ |
| Subset search cost | – | O(2^d) | – | combinatorial | O(d²) params, L-BFGS |
| Scales to d ≥ 100 on one GPU in seconds | ✓ | – | ✓ | – | ✓ (d=200, ~24 s) |
| OOD: mechanism shift (empirical) | fails | ✓ | ~ | n/a | ✓ (== oracle) |

## 3. Articulation to avoid overclaiming

1. **Do NOT claim** a fundamentally new theory of causal representation
   learning; the linear core composes known identifiability results
   (equal-variance) with known multi-env heterogeneity logic
   (Peters et al.) in a new, differentiable, scalable optimizer.
2. **DO claim** the T2 counterexample (pooled DAG learning is non-identifiable
   for the invariant subgraph) — this inversion is genuinely useful framing.
3. **DO claim** the neural SCIN negative result as a finding: current
   encoder→latent-SEM pipelines destroy the invariance signal (IRMs-divergence
   in latent space), which supports the broader thesis that *invariance into
   deep representations remains unsolved*.
4. Benchmarked baselines must include at minimum: NOTEARS-pooled,
   NOTEARS-per-env-then-intersect, ICP (via the `causal-learn` implementation),
   V-REx, and ERM. (V-REx and NOTEARS-pooled are done; ICP subset-search at
   d=10 is feasible on CPU and is the one remaining head-to-head to add.)

- [x] Real-data validation: Causal Chambers light tunnel (lt/standard,
      Gamella et al. 2025) with physical ground truth — SCIN precision 0.79
      vs pooled NOTEARS 0.26, F1 0.430 vs 0.300, SHD 29 vs 61 (Exp G)

## 4. Status

- [x] Gap analysis drafted (this file)
- [x] Add ICP (own node-wise subset solver, exact F-tests under A1) baseline
      to Exp A at d=10 — 2 envs: F1 0.107±0.068; 4 envs: 0.117±0.070 vs SCIN
      0.945 — subset-intersection collapses in this heterogeneity regime
      (results/exp_f_icp_baseline_nenvs*.json, fig5_icp.png)
- [ ] Related Work prose for the paper itself (after T1/T3 proofs are final)
