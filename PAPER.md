# PAPER.md — Draft skeleton (v0.1)

**Working title:**
*"Sparse Causal Intervention Networks: Learning Invariant Mechanisms
from Multiple Environments for Distribution-Shift Generalization"*

**Target:** Nature Machine Intelligence (first choice); fallback Nature
Computational Science. Short-format (NMI Letter ~4,500 words / Article
~3,000–5,000 words main text, Methods after).

**Claim discipline (per plan v2.0):** every number below is committed in
`results/*.json` + `RESULTS.md`; overclaiming is fenced by the "Limits"
lines in each section. The neural and full-chambers negative results are
part of the paper, not hidden.

---

## Abstract (≤ 200 words, to write last)

Skeleton points:
- Problem: predictive models fail under distribution shift because they
  fit environment-dependent (spurious) mechanisms alongside invariant
  causal ones.
- Approach: SCIN — one differentiable objective that jointly fits a
  shared mechanism matrix W plus per-environment deltas Δ_e, uses
  per-environment acyclicity for structure and an equal-variance
  criterion for orientation, then selects edges whose coefficients are
  invariant across environments.
- Killer numbers: invariant-edge recovery F1 0.945±0.028 vs 0.851 for
  pooled continuous DAG learning (synthetic); OOD error equal to the
  oracle noise floor (0.99 vs ERM 10.1, 10×) under a strong spurious
  shift; precision 0.69 vs 0.30 on a physical system with validated
  ground truth (Causal Chambers light tunnel); beats a node-wise ICP
  baseline (F1 0.94 vs ~0.11). Theory: identifiability under
  equal-noise variances + quantitative mechanism heterogeneity, and an
  exact OOD decomposition showing pooled ERM pays a shift-amplified
  spurious term that SCIN provably removes.
- Honest boundary: linear mechanisms; the neural instantiation and the
  full 38-node chamber graph are documented negative results.

---

## Main text (Results-first, Nature format)

### 1. Introduction (≈ 800 words)

- The distribution-shift failure is a mechanism-inference problem:
  models trained on pooled data entangle invariant causal mechanisms
  with environment-dependent ones (cite: Pearl 2009; Peters et al. 2016;
  Arjovsky et al. 2020 IRM; Kilbertus et al.).
- Existing options fail at the margins: ICP-type invariance *testing*
  has no statistical power at weak, sparse heterogeneity (we quantify:
  F1 ≈ 0.11); pooled continuous DAG learning cannot in principle
  separate invariant from moved mechanisms (T2 here: a constructive
  two-SCM indistinguishability); IRM/V-REx shift invariance to the risk
  level, which is too coarse when a spurious feature lowers risk in all
  training environments (Exp B: V-REx ≈ ERM).
- Contribution (3 bullets):
  1. SCIN: joint continuous estimation of {invariant mechanisms,
     moved mechanisms, orientation} from 2+ environments via a single
     differentiable objective (per-env acyclicity + equal-variance
     criterion + invariance selection).
  2. Theory: identifiability with quantitative heterogeneity
     (A2∗, Theorem 1), impossibility for pooled rules (T2), exact OOD
     decomposition with a selection-case analysis that predicts both
     the wins and the failure modes (T3/T3′).
  3. Empirics spanning synthetic → physical: 5 falsification
     experiments where two are honest negatives; a sensitivity analysis
     that maps the selection window rather than cherry-picking.

### 2. Results (≈ 2,000 words; all figures here)

#### Results 2.1 — Invariant-mechanism recovery: precision is the battleground
Fig 1 (exp A table + Phase-2 ablation panel).
- SCIN F1 0.945±0.028 vs NOTEARS-pooled 0.851 vs per-env intersection
  0.839; the entire advantage is precision (+0.16–0.18).
- Ablation: equal-variance criterion most load-bearing (0.945→0.244
  without it); invariance statistic and per-env deltas each carry
  ~0.11–0.13. Every component earns its place.

#### Results 2.2 — Scaling to d = 200 on GPU with flat runtime
Fig 2 (fig1_scaling.png).
- F1 0.889/0.928/0.905/0.958 at d = 25/50/100/200 (n=2000/env,
  5-seed CPU + RTX PRO 6000 Blackwell GPU runs identical concept;
  precision 0.96–1.00, SHD 5–18, runtime 23 s at d=200, fp64).
- Sample-complexity: flat F1 from n=250/env at d=10.

#### Results 2.3 — OOD generalization: oracle-equal under spurious shift
Fig 3 (fig3_ood_bars.png).
- Exp B (child-side spurious, s←y): SCIN == Oracle exactly
  (10/10 seeds), OOD MSE 0.994 vs ERM 10.125 (10×), ID penalty 2.2×
  (report it: invariance trade-off is real).
- Exp E (confounded, s→y INSIDE the mechanism, sign-flip shift):
  SCIN 1.006±0.015 == Oracle 1.005 at the final setting; ERM 1.339.
  Selection = {x0,x1,x2} in 5/5 seeds.
- V-REx ≈ ERM here — risk-level invariance does not see the mechanism.
- Selection-sensitivity (fig6_e_sens.png): robust plateau exists —
  var_tol∈[0.02,0.05]×tau∈[0.05,0.1]; both failure modes at the
  extremes are predicted by T3's case analysis (under-selection drops
  a parent; over-selection re-admits s). No cherry-picking: the grid is
  in the paper.

#### Results 2.4 — The baseline class that cannot work: ICP has no power
Fig 4 (fig5_icp.png).
- Node-wise ICP with exact F-tests: F1 0.107±0.068 (2 envs) /
  0.117±0.070 (4 envs) vs SCIN 0.945/0.946 on identical data.
- Mechanism: subset-intersection receives ~350/512 accepted subsets
  per target; empty intersections in 7–10/10 targets.
- Framing (non-competitor): invariance testing (ICP) and invariance
  estimation (SCIN) occupy opposite regimes; complementary, not rival.

#### Results 2.5 — Physical system: Causal Chambers light tunnel
Fig 5 (new: Exp G adjacency comparison — TO MAKE; currently tables only).
- 12 real interventions, 18-node observable subgraph, 37 GT edges.
- SCIN precision 0.786 (CPU) / 0.688 (GPU 5-seed) vs pooled 0.263/0.300;
  F1 0.430/0.415 vs 0.300/0.345; SHD 29/31 vs 61/57. Best tau = the
  a-priori default 0.1 — hyperparameters transfer from synthetic. No
  retuning.
- Precision-limited story: what SCIN selects is mostly physically real;
  what it misses is explained (unmeasured diode-internal parents,
  cos²-polarizer nonlinearity, n=500).

#### Results 2.6 — Honest boundaries (reviewers trust these; Nature demands them)
- **Part B, full 38-node chamber graph, 33 envs: NEGATIVE.** SCIN F1
  0.161 vs pooled 0.383 — but pooled "wins" by overfitting (113 SHD vs
  57 GT edges, precision 0.28). Failure is upstream of the selection
  rule: parents varying in 1/33 envs have unidentifiable coefficients;
  sin/cos mechanisms are not locally linear; multicollinear light basis.
  Ties directly to T3′ non-coverage list.
- **Neural SCIN: NEGATIVE across 3 design generations** (random
  encoder / ERM-pretrained encoder / joint end-to-end AL training;
  5 seeds each on GPU). Best variant OOD 0.436±0.054 vs ERM-MLP
  0.383±0.019. Diagnosis: encoder entangles the spurious direction with
  causal latents; the invariance statistic in latent space is fragile.
  Future work: invariance-aware encoder training.
- **PNL robustness (partial positive):** F1 0.930→0.856 as tanh
  distortion goes 0→1.0; graceful, no cliff (fig2_pnl.png).

### 3. Discussion (≈ 1,000 words)

- Invariance is a *statistic on coefficients*, not on risks — the
  conceptual takeaway (connects to why IRM fails here).
- Testing vs estimation dichotomy (ICP vs SCIN).
- When pooled baselines "look better": overfit-structure accounting
  (SHD accounting) as a reporting practice recommendation.
- Broader impact: high-precision partial recovery is the safe operating
  mode for scientific discovery — a conservative edge set with 0.79
  precision is actionable for experiment designers (chambers case);
  a dense 0.28-precision graph is not.
- Limitations: linear + equal-variance identifiability (degrades
  gracefully: fig4 panel b), A3 is an assumption (T3′), solver genie
  gap in T1 (flagged, standard in the NOTEARS literature).
- Future: invariance-aware representation learning; rarely-varying
  parents; nonlinear mechanism classes (PNL-robust continuous
  orientation criteria).

### 4. Methods (≈ 3,000 words, after main text)

1. The SCIN objective (per-env W_e = W + Δ_e; per-environment
   polynomial acyclicity tr((I+W∘W/d)^d)−d; ℓ1 on W; ridge on Δ_e;
   equal-variance penalty in original units; augmented Lagrangian,
   L-BFGS strong-Wolfe inner solver, fp64).
2. Selection statistic + masked variant (per-parent env subsets with
   per-env std > 0.1, ≥3 varying envs).
3. Theory (condensed T1–T3; full proofs Supplementary, THEORY.md v2.0).
4. Experimental protocols: Exp A/B/E/E-sens/F/G synthetic + chambers
   setup; environment counts, seeds, HP selection on validation seeds
   only (never test); GPU runs (RTX PRO 6000 Blackwell 96 GB; Colab
   replication 5 seeds).
5. Baselines: NOTEARS-pooled, NOTEARS-per-env-intersection, V-REx,
   node-wise ICP (exact subset enumeration + F-tests), correlation
   thresholding, Oracle.
6. Statistics: per-seed mean±sd, paired comparisons, effect size
   (Cohen's d ≈ 22 for Exp B headline); all raw JSONs in repo.
7. Reproducibility: single-cell Colab scripts + marimo notebook,
   SMOKE flags, commit-pinned; figure-generation script.

### Supplementary

- Full theorem statements + proofs (THEORY.md v2.0 verbatim).
- All result JSONs and logs; negative-result appendix (neural SCIN
  3 generations; Chambers Part B failure-mode analysis).
- Bug-and-fix log (serves reproducibility culture; e.g. hstack
  interleave, fp32 L-BFGS overflow, standardized-units ev bug).

---

## Figure plan (final)

| # | Content | Status |
|---|---|---|
| Fig 1 | Concept: mechanism moves + estimator schematic (TO DRAW) | TODO |
| Fig 2 | Scaling (d → 200 GPU; n/env) | done (fig1) |
| Fig 3 | OOD bars: Exp B + Exp E vs oracle floor | done (fig3) |
| Fig 4 | Ablations + A1-violation panel | done (fig4) |
| Fig 5 | ICP power collapse | done (fig5) |
| Fig 6 | Selection-sensitivity plateau | done (fig6) |
| Fig 7 | Chambers: adjacency heatmaps GT/SCIN/pooled (NEW) | TODO |
| Fig 8 | PNL curve | done (fig2) |

Remaining figure TODOs: Fig 1 schematic, Fig 7 adjacency comparison
(requires storing fitted matrices in exp_g scripts — small code change,
CPU-runnable).

---

## Writing-status checklist

- [x] Theorems T1/T2/T3 full proofs (THEORY.md v2.0)
- [x] All experiments committed with raw JSONs + figures
- [x] Related-work gap analysis (RELATED_WORK.md)
- [ ] Fig 1 schematic + Fig 7 adjacency comparison
- [ ] Abstract + Introduction prose
- [ ] Results prose (from section skeletons above)
- [ ] Methods prose
- [ ] Cover letter (NMI)
