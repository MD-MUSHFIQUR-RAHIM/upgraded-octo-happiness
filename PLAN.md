# SCIN — Sparse Causal Intervention Networks

**Research Plan v2.0** (supersedes v1.0 dated 2026-11-18)
*Working title:* "Causal Representation Learning via Sparse Mechanistic Graphs: Toward AI Systems That Generalize Across Distribution Shifts"

---

## What changed from v1.0 and why

| v1.0 | v2.0 | Reason |
|---|---|---|
| 3 modalities (NLP + vision + genomics) in 12 months | **Core: synthetic + one real domain.** Genomics = stretch goal / second paper | 3 modalities + 3 theorems in 12 months is not credible for one lead author; reviewers will read overreach |
| Theorem 3: "any sufficiently expressive net converges to causally disentangled representation" | **Dropped.** Replaced with a conditional identifiability result **+ a non-identifiability counterexample** | T3 as stated is false in general; a counterexample paper-strengthens the positive result |
| Theorem 2 OOD bound assumed identified causal variables are correct | **Restated with explicit assumption** (A3 below) and a discussion of when A3 fails | Hidden assumptions are the first thing reviewers find |
| Gap analysis cited Pearl 2009 / IRM 2020 / Elhage 2021 | **Literature refresh mandated for 2023–2025**: causal representation learning (von Kügelgen et al. 2024; Brehmer et al. 2022; Ahuja et al. 2023), causal circuit analysis in LLMs, SAE-based interpretability | The field moved; a 2026 submission citing 2021 interp is desk-reject risk |
| ~1200 A100-hours | **~350 A100-hours** phased behind go/no-go gates | Compute follows evidence, not the reverse |

---

## Core hypothesis (sharpened)

> **H1:** Neural networks trained across heterogeneous environments encode *mechanism-level* structure that can be exposed as a sparse, differentiable DAG, and the *invariant subgraph* of that DAG predicts which features the model may safely rely on under distribution shift.

**H0 (null):** Invariance-based edge pruning recovers no more of the invariant mechanism than naive per-environment fitting + intersection. *(This is the baseline to beat — see Experiment A.)*

**Explicit assumptions (stated up front, not buried):**
- **A1 (equal error variances):** linear-Gaussian SEMs with equal noise variance are DAG-identifiable (Peters & Bühlmann 2014). All synthetic experiments satisfy A1; violations are studied in the robustness analysis.
- **A2 (mechanism heterogeneity):** at least one edge mechanism differs across training environments, and the invariant mechanisms of the target are shared.
- **A3 (invariance ⇒ relevance):** features whose incoming mechanisms are invariant and which are ancestors of the target are exactly the features safe to use OOD. **A3 is an assumption, not a theorem — the paper's identified-limitations section is built around when A3 fails** (e.g., anti-causal tasks, latent confounding).

---

## Theorems (restated, defensible)

- **T1 (Identifiability, conditional).** Under A1, A2, faithfulness, and λ₃ scaling as λ₃ = λ₀√(log d / n), the SCIN estimator recovers the invariant subgraph of the true DAG with probability → 1 as n → ∞.
- **T2 (Non-identifiability counterexample).** There exist two SCMs with the same observational distribution across all environments but different invariant subgraphs. ⇒ invariance alone cannot identify causal structure without A1-style restrictions. *(This result protects T1 from over-claiming and is a genuine contribution.)*
- **T3 (OOD risk, conditional on A3).** If A3 holds and the selected feature set S contains the true causal parents, then R_OOD(f_S) ≤ R_ID(f_S) + O(√(|S| / n)) via standard uniform-convergence arguments; the contribution is showing the SCIN selection error enters only through a P(selection ≠ S⁎) factor.

---

## Experiments

### A. Synthetic: invariant-subgraph recovery (GO/NO-GO GATE — this repo)
- d = 10, Erdős–Rényi DAG, ~10 edges, equal noise variances (A1), 2 environments, 3 edge mechanisms perturbed in env 2.
- **Ground truth:** the invariant subgraph (edges with identical mechanism in both envs).
- Methods: (1) correlation thresholding, (2) NOTEARS pooled, (3) NOTEARS per-env + intersection, (4) **SCIN**.
- Metrics: directed F1, precision, recall, SHD; 5+ seeds; thresholds tuned on a *held-out validation seed*, never on test seeds.
- **Gate:** SCIN must beat NOTEARS-per-env-intersection on invariant-subgraph F1 by ≥ 0.10 with 5 seeds. If not, the joint-training claim is dead and the project pivots to the pure interpretability angle.

### B. Synthetic OOD: spurious-correlation removal (this repo)
- 7-node SCM: x0→x1, x0→x2, x1→y, x2→y (invariant), y→s with environment-dependent strength β ∈ {2, 1} (train), β ∈ {0, −2} (test).
- Methods: ERM (all features), IRM, Oracle (true parents), **SCIN** (structure-learn → select ancestors of y in invariant subgraph → retrain predictor).
- Metrics: ID accuracy, OOD accuracy at two shift strengths, ID−OOD gap; 10 seeds; paired Cohen's d vs ERM.

### C. Real data (post-gate, GPU phase)
1. **Causal Chambers** (real physical testbed, published ground truth) — primary real validation.
2. **Single-cell perturbation seq (Replogle/Norman2019)** — stretch goal; only if A+B pass and a computational-biology collaborator is onboard.

### D. Ablations (all mandatory)
Remove DAG constraint / sparsity / invariance loss / per-env deltas; each must justify itself or be cut from the method.

---

## 12-month timeline (gated)

| Months | Work | Gate |
|---|---|---|
| 1–2 | Literature refresh (2023–25), hypothesis doc, this synthetic harness | — |
| 3–4 | SCIN full implementation; Exp A + B at scale; **GO/NO-GO** | Exp A gate (F1 gap ≥ 0.10) |
| 5–7 | Causal Chambers + one large-scale domain; T1/T2/T3 proofs | Exp C real-data positive result |
| 8–9 | Full ablations, robustness (violating A1), statistics | — |
| 10–12 | Writing, figures, code release, submission to **Nature Machine Intelligence** (Nature only if genomics/Chambers story is strong) | — |

## Compute budget (revised)
- Months 1–4: CPU / single GPU — **0 A100-hours** (this repo runs on CPU).
- Months 5–9: ~350 A100-hours (~$500–900 on Lambda/vast.ai), released only after the Month-4 gate.

## Failure modes carried over from v1.0 (still binding)
No "SOTA" framing · negative results reported · ≥5 seeds + effect sizes · code/data availability mandatory · results before narrative.

---

## Repository layout

```
scin/            library (SCM generation, NOTEARS baseline, SCIN model, metrics)
experiments/     exp_a_graph_recovery.py, exp_b_ood.py
results/         JSON results + generated summary tables
```

Run:
```bash
python experiments/exp_a_graph_recovery.py
python experiments/exp_b_ood.py
python experiments/summarize.py
```
