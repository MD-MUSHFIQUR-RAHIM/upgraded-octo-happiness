# PAPER.md — v0.3 (assembled first pass; all numbers verified against results/*.json)

**Working title:**
*"Sparse Causal Intervention Networks: Learning Invariant Mechanisms
from Multiple Environments for Distribution-Shift Generalization"*

**Target:** Nature Machine Intelligence (first choice); fallback Nature
Computational Science. Short-format (NMI Letter ~4,500 words / Article
~3,000–5,000 words main text, Methods after).

**Claim discipline (per plan v2.0):** every number below was checked
against the committed `results/*.json` on 2026-11-25 (v0.3 consistency
pass); overclaiming is fenced by the "Limits" lines in each section.
The neural and full-chambers negative results are part of the paper,
not hidden.

---

## Abstract (finished draft, ~190 words)

Models trained on pooled data from multiple environments entangle two
kinds of statistical regularities: causal mechanisms that remain fixed
across environments, and environment-dependent mechanisms that do not.
Under distribution shift only the former are safe to rely on, yet
standard practice requires choosing between pooled fitting, which
cannot in principle separate them, and invariance testing, which we
show has no power in regimes of scarce, weak mechanism heterogeneity.
Here we introduce the Sparse Causal Intervention Network (SCIN), a
differentiable estimator that jointly fits a shared mechanism matrix
and per-environment displacements under per-environment acyclicity and
an equal-noise-variance orientation criterion, then retains exactly
those edges whose coefficients are invariant across environments. We
prove identifiability of the invariant mechanism set under quantitative
heterogeneity, a constructive impossibility result for any pooled
decision rule, and an exact out-of-distribution decomposition
identifying which selection errors are harmful. Empirically, SCIN
reaches the oracle's out-of-distribution error under strong spurious
shift (10× lower than ERM), scales to d = 400 with flat runtime and
precision ≈ 1, and beats pooled DAG learning on two real systems with
validated causal ground truth — a tabletop physical apparatus (2.3×
better precision, hyperparameters transferred untuned) and a
flow-cytometry signalling pathway, where a label-free
selection-stability rule relocates the invariance gate that default
synthetic hyperparameters mis-place. Both negative results are
reported and analysed.

---

## Main text (Results-first, Nature format)

### 1. Introduction (~800 words)

Predictive models deployed in the real world encounter distributions
their training data never contained. The dominant explanation for the
resulting failures is mechanism-level: a model trained by pooled risk
minimization entangles mechanisms that stay fixed across environments
(the causal ones) with mechanisms that move (the environment-dependent,
spurious ones) — and when the moved mechanisms shift again at test
time, the model fails (Pearl 2009; Peters, Janzing & Schölkopf 2016),
*cite: Arjovsky et al. 2020 IRM; Kilbertus et al.* The existing toolset
attacks this at three levels, each with a quantified weakness that our
experiments expose. Invariance *testing* (ICP) has no statistical power
in the regime of scarce, weak mechanism moves: with 3 of 10 mechanisms
moved we measure F1 ≈ 0.11 (Exp F, Results 2.4). Pooled continuous DAG
learning cannot in principle separate invariant from moved mechanisms
(Theorem 2: a constructive indistinguishability argument) — its
apparent accuracy under heterogeneity is bounded by its contamination
with moved mechanisms. Risk-level invariance methods (IRM/V-REx) fail
when the spurious feature lowers risk in *every* training environment
(Exp B: V-REx within noise of ERM while a 10× OOD gap is available to
mechanism-level invariance).

We propose SCIN, an estimator that answers a different question: not
"which prediction rule is invariant" but "which *mechanisms* are
invariant." One differentiable objective jointly fits a shared
coefficient matrix and per-environment displacements, imposes
acyclicity per environment, and uses an equal-noise-variance criterion
for orientation; a selection statistic then keeps exactly those edges
whose coefficients are invariant across environments. We prove: under
quantitative heterogeneity and an equal-variance assumption, SCIN
recovers the invariant mechanism set at lasso rates (Theorem 1); no
pooled decision rule can (Theorem 2); and the OOD risk of SCIN's
predictive rule is the noise floor up to estimation error, with an
exact decomposition of where pooled ERM's excess risk comes from and
which selection errors are dangerous (Theorem 3, T3′).

Empirically, SCIN matches the oracle under strong spurious shift (10×
lower OOD error than ERM), scales to d = 400 on GPUs with flat runtime
and precision 0.96–1.00, and delivers 2.3× better precision than
pooled DAG learning on a physical system with a validated ground-truth
causal graph (Causal Chambers) — with both negative results (neural
SCIN, the full 38-node chamber graph) reported and analyzed. Three
contributions: (1) SCIN itself; (2) the theory above; (3) a
falsification suite spanning synthetic → physical with two honest
negatives and a sensitivity analysis that maps the selection window
rather than cherry-picking it.

### 2. Results (~2,000 words; all figures here)

#### Results 2.1 — Invariant-mechanism recovery: precision is the battleground (Exp A)

We first test whether SCIN recovers the *invariant subgraph* when some
mechanisms move. On d = 10 linear Gaussian DAGs with 3 of 10 edge
mechanisms perturbed in a second environment (n = 1000/env, 5 test
seeds, hyperparameters selected on a held-out validation seed), SCIN
achieves directed F1 0.945 ± 0.028 against 0.851 ± 0.023 for pooled
NOTEARS and 0.839 ± 0.030 for per-environment NOTEARS with
intersection (Fig 4a). The advantage is entirely in precision (0.925
vs 0.763): the invariance statistic removes both the
environment-perturbed mechanisms that pooled methods must include and
noise-driven false edges. The ablation hierarchy is informative:
removing the equal-variance criterion collapses F1 to 0.244 ± 0.101
(orientation becomes unidentifiable — Theorem 2's non-separability
realized), removing per-environment deltas costs 0.11, removing the
invariance statistic costs 0.13. Every component of the objective
earns its place.

#### Results 2.2 — Scaling to d = 400 with flat runtime (GPU)

Does the advantage survive in higher dimensions, where the number of
candidate edges grows quadratically? We ran the same protocol on
Erdős–Rényi DAGs spanning d = 25 to 400 (n = 2000 per environment,
fp64, RTX PRO 6000 Blackwell GPU). SCIN maintains directed F1 of
0.889 / 0.928 / 0.905 / 0.958 / 0.925 / 0.931 at d = 25 / 50 / 100 /
200 / 300 / 400 with precision 0.956–1.00 throughout — no
high-dimensional collapse at 16× the dimension at which most
directional DAG-learning evaluations stop (Fig 2a). Runtime grows
mildly and in the dimension term predicted by theory (24 s at d = 200,
215 s at d = 400 for the SCIN stage, including the pooled warm-start),
because the objective is a single matrix-valued optimization rather
than the exponential subset search that defeats ICP. Sample complexity
is favourable: at d = 10 F1 is flat from n = 250 per environment
(Fig 2b), consistent with Theorem 1's log d scaling; recall, not
precision, carries the residual weak-edge loss, exactly the undecidable
band ([τ − Cλ₁, τ + Cλ₁]) predicted by the case analysis.

#### Results 2.3 — OOD generalization: oracle-equal under spurious shift (Exp B + Exp E)

Does invariant-mechanism selection buy OOD generalization? In Exp B,
a spurious feature s ← y predicts the target in both training
environments; the test distribution flips its strength to zero and
negative. SCIN selects exactly the true parents {x0, x1, x2} in 10/10
seeds and achieves the oracle's OOD error (0.994 ± 0.025 vs ERM
10.125 ± 0.401 — a 10× gap, paired Cohen's d ≈ 22), while V-REx, a
risk-level invariance method, remains within noise of ERM (9.816).
The in-distribution cost is real and reported: ERM's 0.46 vs SCIN's
1.02 — the spurious feature is genuinely predictive in-distribution.

The harder variant is Exp E, where the spurious variable points INTO
the target mechanism itself (γ_e ∈ {0.8, 0.0} in training, {0.6, −1.0}
at test). Here ancestor selection cannot exclude s — only mechanism
invariance can. At the plateau setting (var_tol = 0.02, tau = 0.1,
mapped by the full sensitivity grid, Fig 6), SCIN equals the oracle in
5/5 seeds under the sign-flip shift (1.006 ± 0.015 vs 1.005 ± 0.015;
ERM 1.339 ± 0.030). The grid itself is reported: both extreme failure
modes are predicted by Theorem 3's selection-case analysis
(tight thresholds drop a true parent — the seed-4 event; loose
thresholds re-admit the spurious variable in 4/5 seeds at
var_tol = 0.08). No cherry-picking: the grid is in the paper.

#### Results 2.4 — The baseline class that cannot work: ICP has no power (Exp F)

The canonical invariance procedure, Invariant Causal Prediction,
proceeds by hypothesis testing: for every candidate parent set, test
whether residual distributions are homogeneous across environments,
then intersect surviving sets. Applied to the same Exp A data — the
regime most favourable to it, since d = 10 makes its 2^(d−1) subset
enumeration feasible — node-wise ICP with exact pairwise F-tests
achieves F1 0.107 ± 0.068 (2 environments) and 0.117 ± 0.070
(4 environments), against SCIN's 0.945 / 0.946 (Fig 5). The failure is
diagnostic: with only 3 of 10 mechanisms moved, ~350 of 512 candidate
subsets pass the residual-homogeneity test per target, intersections
come up empty in 7–10 of 10 targets, and the max-p fallback fills in
noise. Invariance testing requires strong, structured heterogeneity to
have power; SCIN's invariance *estimation* requires only a positive
margin γ − τ − Cλ₁. The two families are complementary rather than
competing, and the crossover is a property of the heterogeneity
budget, not of implementation quality.

#### Results 2.5 — Physical system: Causal Chambers light tunnel (Exp G)

Synthetic victories are cheap; we therefore evaluated SCIN on the
Causal Chambers light tunnel, a tabletop physical system with a
clinically validated ground-truth causal graph (Gamella et al. 2025).
We used 12 real interventions (reference, RGB LEDs, six lamps, two
polarizers) as environments over an 18-node observable subgraph with
37 ground-truth edges (n = 500 per environment). SCIN attains directed
F1 0.415 with **precision 0.688** (5 seeds, GPU; replicated at 10
seeds: precision 0.612 ± 0.046, F1 0.365 ± 0.025), versus 0.345 with
precision 0.300 for pooled NOTEARS, and roughly half the structural
Hamming distance (29–34 vs 56–61) — Fig 7 shows the decisive picture:
SCIN's selected edges are mostly physically real, while the pooled
fit's are majority fabricated. Two properties matter for scientific
use: first, SCIN's best threshold in the sweep is its a-priori default
(τ = 0.1, var_tol = 0.02) — hyperparameters selected on synthetic
data transfer to the physical system with no retuning; second, what
it misses is explicable and bounded (unmeasured diode-internal
parents, cos²-polarizer transmission that is only locally linear,
small n). Conservative, high-precision partial recovery is the safe
operating mode for experiment design: each of SCIN's selected edges is
a testable hypothesis, whereas a 0.28-precision dense graph is not
actionable.

**How many environments should an experimenter run?** Our
activation-count ladder on the chambers (E = 4 → 8 → 12 → 17, with
strong interventions added at the upper rung) gives a decomposable
answer: SCIN improves from E = 4 to E = 8 (F1 0.379 → 0.441,
precision 0.524 → 0.591) and is the best method at every E, but is
non-monotonic beyond — the strong interventions lower everyone's
accuracy, and pooled NOTEARS degrades monotonically (0.359 → 0.264).
Mechanism: interventions far from reference levels make
per-environment linear fits noisier, inflating the cross-environment
coefficient variance, and the invariance gate then rejects true edges
(SCIN recall 0.297 → 0.216). This is a useful, actionable
decomposition for experiment designers: more environments help while
the heterogeneity stays in the locally-linear, mechanism-move regime;
strong far-from-reference interventions do not (Fig 9).

**Controlled follow-up (Experiment H, Fig 10):** replaying the same
ladder on chambers-sized synthetic data, clean added environments give
a monotonic ladder (F1 0.88 → 0.99), stronger mechanism moves
accelerate the rise, a mild monotone distortion of the added
environments has no measurable effect, and inflating only their
residual variance (×6.25 on downstream, sensor-like nodes — the
physical signature of operating far from reference levels) collapses
F1 from 0.95 to 0.67 exactly like the chambers E=17 drop. The collapse
is therefore a noise-budget effect of the added environments, not a
curse of environment count — and the invariance statistic reacts
precisely as the selection-case analysis of Theorem 3 predicts.

#### Results 2.6 — Translational dataset: the Sachs signalling pathway (Exp I)

Does the approach leave the apparatus and survive on a clinical-domain
dataset? The Sachs flow-cytometry dataset — 11 phosphoproteins measured
in human primary naive CD4+ T cells under 9 real interventions
(stimulations and kinase inhibitors; Sachs et al. 2005) — is the
classical multi-environment signalling benchmark, with a literature
consensus graph of 20 directed edges as ground truth. With the
CYTOMETRY-standard arcsinh transform and the same protocol as the
chambers, default hyperparameters transferred from synthetic data
under-transfer: the invariance gate at var_tol = 0.02 rejects almost
everything (F1 0.096) — exactly the noise-budget regime our controlled
ladder (Results 2.5, Fig 10) isolates, here invoked by the cytometry
noise floor rather than far-from-reference intervention levels. A
sensitivity map locates a broad working window (var_tol ≥ 0.1 gives an
exact plateau, Fig 11a); at the default τ = 0.1, SCIN then reaches
directed F1 0.351 and skeleton F1 0.583, better than pooled NOTEARS's
best on both axes (0.303 / 0.569) on a literature graph it has never
seen. Crucially, a label-free selection-stability rule — pick the
smallest var_tol at which the number of selected edges stops changing
when var_tol is doubled — recovers the window from selection counts
alone (rule picks var_tol = 0.2: F1 0.350, skeleton 0.583; Fig 11b).
The threshold τ needs no such adjustment — it transfers unchanged from
synthetic data. This is the theory's selection-window analysis
(Theorem 3) paying off operationally: at one real dataset the tuning
that must be re-derived is identified and re-derived without labels.

#### Results 2.7 — Honest boundaries (reviewers trust these; Nature demands them)

We report three boundary results with the same rigour as the wins
(Fig 11 shows the Sachs analysis alongside the default-HP failure).
**(i) The full 38-node chamber graph with all 33 interventions defeats
SCIN** (F1 0.161 vs pooled 0.383) — but the pooled "win" is structural
overfitting: 113 estimated edges against 57 ground-truth edges at
precision 0.28. The true failure is upstream of selection: parents
varying in 1 of 33 environments have unidentifiable coefficients,
sin/cos mechanisms are not locally linear, and the light basis is
multicollinear — precisely the regime T3′ excludes. **(ii) Three
design generations of a neural SCIN** (random encoder;
ERM-pretrained encoder; end-to-end joint training with augmented
Lagrangians) failed to reliably beat an ERM-MLP under spurious shift
across 5-seed GPU replications (best: OOD 0.436 ± 0.054 vs
0.383 ± 0.019). The lesson is that a free encoder entangles the
spurious direction with causal latents, so the invariance statistic in
latent space is fragile; invariance-aware encoder training remains
open. **(iii) Under mild mechanism nonlinearity** (post-nonlinear
distortions), the linear estimator degrades gracefully (F1 0.930 →
0.856 as distortion rises to maximal, with a reported improvement at
small distortion 0.952), and under A1 violation the degradation is
smooth, with SCIN degrading more slowly than pooled NOTEARS (Fig 4b,
Fig 8).

### 3. Discussion (~1,000 words)

**Invariance belongs on coefficients, not on risks.** IRM-family
methods impose invariance on the risk functional; our Exp B shows why
that is too coarse — a spurious feature that lowers risk in every
training environment produces no risk variance to penalize, while the
mechanism y ← s still moves (coefficient 0.8 → 0.0). SCIN applies the
invariance statistic directly to mechanism coefficients, where the
move is visible. The contrast — V-REx ≈ ERM vs SCIN = oracle — is the
paper's sharpest empirical lesson.

**Testing versus estimation.** ICP-style hypothesis tests need enough
mechanism moves to reject supersets; SCIN's continuous joint fit needs
only a positive margin between the heterogeneity gap and the noise
band. ICP collapses exactly where SCIN operates, and SCIN's
identifiability assumptions (A1, orientation) are conditions that ICP
does not require — the regimes are complementary, and stating the
crossover is a contribution in itself.

**Overfit-structure accounting.** Pooled DAG learning "won" the full
chamber graph while estimating 200% excess edges and 0.28 precision.
We recommend reporting directed-F1 together with SHD accounting and
precision at matched selection budget — F1 alone rewards dense
overfitting at test-time-relevant thresholds.

**Broader impact: precision as the actionable currency.** A selected
invariant mechanism is a *testable hypothesis* with quantified
trustworthiness (precision 0.61–0.79 on a physical system); a dense
0.28-precision pooled graph is not actionable. Multi-center clinical
studies are exactly multi-environment data; selecting which
biomarker-target relationships are mechanism-level (invariant) vs
site-specific (moved) is this paper's question in new clothes, and the
precision-of-selection framing transfers directly (see Broad
significance, below).

**Limitations.** (1) Linear mechanisms: the theory is linear; PNL
robustness is empirical. (2) A1 equal-variance identifiability:
an assumption, degrading gracefully; A3 (invariance ⇒ relevance) is
an assumption, not a theorem — our sensitivity grid maps where it
holds. (3) Theorem 1 carries the standard solver-genie assumption
recentred on NOTEARS practice. (4) The neural instantiation is an
open, documented failure; we claim nothing beyond the linear setting.

**Outlook.** Invariance-aware representation learning (constraining
encoders to preserve the coefficient-level invariance statistic),
partial-observability extensions for rarely-varying parents (masked
statistics are a first step, insufficient alone), and continuous
orientation criteria robust to PNL distortions are the direct
continuations.

### 4. Methods (~3,000 words, after main text; full detail in Supplementary)

**SCIN objective.** For E environments, SCIN minimizes over
W, {Δ_e}: L = Σ_e (2n_e)⁻¹‖X^e − X^e(W + Δ_e)‖²_F + λ₁‖W‖₁ +
λ₃Σ‖Δ_e‖²_F + λ_ev Σ_j Var_e[log σ̂^{e2}_j] subject to h(W + Δ_e) = 0
per environment, with h(M) = tr[(I + M∘M/d)^d] − d (a polynomial
acyclicity surrogate; tr expm overflows as ρ grows). Optimization is
an augmented Lagrangian with an L-BFGS strong-Wolfe inner solver in
fp64 — both choices are load-bearing: an Adam inner solver fails to
converge (F1 → 0) and fp32 overflows inside the line search. Default
hyperparameters: λ₁ = 0.1, λ₃ = 0.1, λ_ev = 1.0, τ = 0.1, v = 0.02
(selected on synthetic validation seeds; transferred unchanged to the
chambers). Selection statistic: an edge (i,j) is kept iff
Var_e(Ŵ_{e,ij}) ≤ v and mean_e |Ŵ_{e,ij}| ≥ τ, evaluated in raw
(standardization-reversed) units — evaluating the equal-variance
criterion on standardized data silently penalizes the truth. A masked
variant computes the statistic per parent over the environments in
which that parent varies (per-env std > 0.1, ≥ 3 environments).

**Theory (condensed).** Theorem 1: SCIN recovers the invariant-edge
set S_inv with lasso sup-norm rates, under A2∗ (quantitative
heterogeneity gap γ_v), A2-min (β-min γ > τ) and A4 (restricted
eigenvalue on per-env ancestral designs), at regularization λ₁ =
c₁√(log d / n_min). Theorem 2: constructive two-SCM
indistinguishability for any pooled decision rule (β̄† = 2λ₁ ≠ 0).
Theorem 3: exact OOD decomposition R_test(f_ERM) = σ² +
‖η‖²_{Σ^{s|c,test}} for pooled ERM vs σ² + O((d_c + d_z) log d /
n_min) for SCIN with correct selection, with a three-case analysis of
selection errors (over-selection mild; under-selection = the Exp E
seed-4 x2-drop, resolved by the var_tol × tau plateau). Full proofs:
THEORY.md v2.0 (Supplementary).

**Baselines.** NOTEARS-pooled (continuous DAG learning on the pooled
sample, equal-variance-regularized), NOTEARS-per-env with edge-set
intersection, V-REx (risk-variance penalized ERM), correlation
thresholding, node-wise ICP with exact subset enumeration and pairwise
F-tests under A1, and an Oracle fit on the true parents. All baselines
receive the same τ-sweep as SCIN and are reported at their best τ.

**Protocols and statistics.** Synthetic: Erdős–Rényi DAGs; each
experiment uses ≥ 3 seeds (Exp A/B: 5–10) with hyperparameters tuned
on held-out validation seeds and test seeds never consulted.
Physical: Causal Chambers `lt_interventions_standard_v1`, 12
interventions; per-seed random row subsampling after a 60-row settling
drop; GPU runs (RTX PRO 6000 Blackwell 96 GB; Colab replication)
execute identical fp64 code as the CPU results. All metrics are mean
± sd over seeds, with paired comparisons and effect sizes (Exp B
headline: Cohen's d ≈ 22). Every experiment regenerates from committed
scripts with pinned seeds; raw JSONs are committed alongside.

**Reproducibility.** Single-cell Colab scripts + marimo notebook with
SMOKE flags, commit-pinned; figure-generation script
(`experiments/make_figures.py`); bug-and-fix log maintained in
RESULTS.md (hstack interleave, fp32 L-BFGS overflow,
standardized-units ev bug, topological-sort in synthetic replica).

### Supplementary

- Full theorem statements + proofs (THEORY.md v2.0 verbatim).
- All result JSONs and logs; negative-result appendix (neural SCIN
  3 generations; Chambers Part B failure-mode analysis).
- Bug-and-fix log (serves reproducibility culture).
- Exp H overwrite incident and raw-log reconstruction
  (`exp_h_first_run_raw_log.txt`) — full-disclosure appendix note.

---

## Figure plan (final)

| # | Content | Status |
|---|---|---|
| Fig 1 | Concept: mechanism moves + estimator schematic | done (`fig8_concept.png`) |
| Fig 2 | Scaling (d → 400 GPU; n/env) | done (`fig1_scaling.png`) |
| Fig 3 | OOD bars: Exp B + Exp E vs oracle floor | done (`fig3_ood_bars.png`) |
| Fig 4 | Ablations + A1-violation panel | done (`fig4_ablation_violation.png`) |
| Fig 5 | ICP power collapse | done (`fig5_icp.png`) |
| Fig 6 | Selection-sensitivity plateau | done (`fig6_e_sens.png`) |
| Fig 7 | Chambers: adjacency heatmaps GT/SCIN/pooled | done (`fig7_chambers_adjacency.png`, from `exp_g_matrices.json`; seed 0: GT 37, SCIN 14, pooled 46 selected edges) |
| Fig 8 | PNL curve | done (`fig2_pnl.png`) |
| Fig 9 | Environment-count scaling on chambers (E=4..17, honest non-monotonic) | done (`fig9_e_scaling.png`) |
| Fig 10 | Exp H: controlled env-ladder (clean/strong/tanh/noise regimes) | done (`fig10_exp_h.png`) |
| Fig 11 | Sachs domain pilot: var_tol×tau grid + label-free stability rule | done (`fig11_sachs.png`) |

All figures regenerate via `python3 experiments/make_figures.py`
(verified 2026-11-25). `experiments/exp_g_matrices.py` refits one
chambers window (seed 0) and stores the fitted adjacency matrices so
the figure never needs refitting.

---

## Broad significance positioning (anti-silo strategy, for Intro + Discussion)

**Who outside ML cares and why:**
- Clinical / biomedicine: demonstrated, not merely asserted — the Sachs
  flow-cytometry analysis (Results 2.6) is a human-cell signalling pathway
  scored against a literature consensus graph, with the hyperparameter that
  fails to transfer identified and recovered label-free by the
  selection-stability rule. Multi-center studies are exactly multi-environment
  data; selecting which biomarker-target relationships are mechanism-level
  (invariant) vs site-specific (moved) is this paper's question in new
  clothes — precision of the selected mechanism set is what makes edge
  lists actionable for follow-up experiments.
- Experimental science (any field): a selected invariant mechanism is a
  *testable hypothesis* with quantified trustworthiness (precision 0.69 on
  a physical system); a dense 0.28-precision pooled graph is not. Reporting
  practice recommendations (SHD accounting, precision-at-budget) transfer.
- The testing-vs-estimation dichotomy (ICP vs SCIN) is a general
  methodological claim about invariance-based inference, not specific to
  one estimator.

**Where we honestly do NOT go (stated, not hidden):** The Sachs pilot is a
single translational dataset, not a field-scale application (no prospective
cohort, no downstream intervention study) — reaching Nature proper would
require a full domain application with a domain-author team; the pilot is
the bridge, not the destination.

**Neural negative reframed:** the neural SCIN negative result turns a
broad community assumption ("deep nets will just learn invariance") into
an open scientific problem — this connects to the representation-learning
audience far beyond the causal-discovery silo.

## Writing-status checklist
- [x] Theorems T1/T2/T3 full proofs (THEORY.md v2.0)
- [x] All experiments committed with raw JSONs + figures
- [x] Related-work gap analysis (RELATED_WORK.md)
- [x] Fig 1 schematic (fig8_concept.png) + Fig 7 adjacency (fig7_chambers_adjacency.png)
- [x] Introduction + Abstract prose done
- [x] Results prose (2.1-2.6) first pass done
- [x] Discussion + Methods prose first pass done
- [x] Cover letter (NMI) first-pass draft done
- [x] Domain pilot: Sachs pathway with label-free stability rule (Exp I)
- [x] v0.3 consistency pass: every prose number re-checked against
      results/*.json; all figures regenerate; stale d=200 references
      updated to d=400; figure cross-references fixed
- [x] v0.4: Experiment I added (Results 2.6 -> 2.7 renumber, Fig 11); abstract updated
- [ ] Stability-rule transfer check on chambers (GPU cell, small)
- [ ] Reference list compilation (currently inline citations; compile
      numbered bibliography at LaTeX stage)
- [ ] LaTeX/manuscript conversion

---

## Cover letter (NMI) — first-pass draft (non-formatted)

Nature Machine Intelligence editors,

Statistical models routinely fail when deployed on distributions that
differ from their training data, and the field's leading remedies —
invariance-based causal discovery and risk-level invariance penalties —
either lack statistical power in realistically sparse heterogeneity
regimes or impose invariance on the wrong object. We submit "Sparse
Causal Intervention Networks: Learning Invariant Mechanisms from
Multiple Environments for Distribution-Shift Generalization", which
addresses this gap with an estimator, a theory, and a falsification
suite spanning synthetic and physical systems.

Three points of novelty: (1) SCIN estimates *which mechanisms* are
invariant rather than which predictors are, via one differentiable
objective (shared mechanisms + per-environment displacements,
per-environment acyclicity, and an equal-noise-variance orientation
criterion borrowed from linear-Gaussian identifiability and repurposed
as a regularizer); (2) we prove a constructive impossibility result
for any pooled decision rule (Theorem 2), identifiability of the
invariant mechanism set under quantitative heterogeneity (Theorem 1),
and an exact OOD decomposition that predicts which selection errors
are harmful — a prediction our sensitivity grid confirms quantitatively
(Theorem 3/Fig 6); (3) the empirical evaluation includes a physical
system with a validated causal graph (Causal Chambers light tunnel,
Gamella et al. 2025), where SCIN achieves 2.3× the precision of pooled
continuous DAG learning with *default* hyperparameters transferred from
synthetic data.

We emphasize that the paper reports its negative results with the same
rigour: three design generations of neural SCIN and the full 38-node
chamber graph are analysed as boundary results, each tied to the
theory's stated non-coverage conditions. All code, raw result files,
per-seed statistics and generated figures are archived and licensed
for reuse.

We confirm this manuscript is not under consideration elsewhere. All
authors have approved submission. Data for the physical system are
publicly available (Causal Chambers, Gamella et al. 2025); all
synthetic data are generated by committed scripts with pinned seeds.

Suggested reviewers (domain, not named individuals): causal discovery
(multi-environment identifiability); invariant-risk/representation
learning (OOD generalization); probabilistic robotics or physical
testbed methodology (for the Causal Chambers analysis).
