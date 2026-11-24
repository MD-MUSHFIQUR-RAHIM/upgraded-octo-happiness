# THEORY.md — Formal Results (Draft v1.0)

Draft statements and proof sketches for the three theorems of the SCIN
paper. These are working drafts: each is checked against the empirical
results (see RESULTS.md), and assumptions are flagged explicitly where
they are assumptions, not theorems.

---

## Setting and Assumptions

**Data-generating process.** For each environment $e \in \mathcal{E}$,
observations $X^e \in \mathbb{R}^d$ follow a linear Gaussian SCM with
weight matrix $W^{e*}$ (acyclic) and independent unit-variance noise:
$X^e_j = X^e W^{e*}_{:,j} + \varepsilon^e_j$,
$\varepsilon^e_j \perp X^e$-past.

**A1 (Equal noise variances, Peters & Bühlmann 2014).** All node noise
variances are equal within each environment (identifiability
assumption).

**A2 (Mechanism heterogeneity).** The perturbed mechanisms differ across
environments: there is a non-empty set $P$ of edges with
$W^{e*}_{ij} \neq W^{e'*}_{ij}$ for some $e \neq e'$, and the
perturbation is *generic*: the induced difference in the fitted
coefficients does not vanish in the population.

**A3 (Invariance ⇒ relevance).** *Assumption, not a theorem.* An edge
that is invariant across all training environments, lies on a directed
path into the target $y$, and has non-negligible weight, is a causal
ancestor of $y$ supported in every future environment. This is the
modularity reading of ICP (Peters et al. 2016).

---

## Theorem 1 (Identifiability of the invariant subgraph)

> **Statement.** Grant A1 and A2. As $n_e \to \infty$ for all $e$, the
> population minimizer of the multi-environment least-squares objective
> with per-environment acyclicity and the equal-variance criterion
> satisfies: (i) each $W^e$ converges to the noise-precision form
> induced by $W^{e*}$; and (ii) the selected invariant edge set
> $\widehat{E}_{\mathrm{inv}}
> = \{(i,j): \mathrm{Var}_e(\hat W^e_{ij}) < v,\ |\bar W_{ij}| \geq
> \tau\}$ equals the true invariant edge set restricted to non-negligible
> edges $E^{\tau}_{\mathrm{inv}}$ — no false additions and no losses —
> with probability $\to 1$.

**Proof sketch.**
1. *Per-environment recoverability.* With Gaussian data and equal noise
   variances (A1), the standardized linear-Gaussian DAG is identifiable
   from $\Sigma^e$ (Peters & Bühlmann 2014, Thm 1). Exact least squares
   over acyclic matrices recovers the noise-precision factorization of
   $(\Sigma^e)^{-1}$; consistency for finite $n_e$ follows from standard
   covariance concentration (Loh & Wainwright 2012-style bounds).
2. *Cross-environment separation.* For $(i,j) \in
   E^{\tau}_{\mathrm{inv}}$, every estimate converges to the same
   constant, so $\mathrm{Var}_e(\hat W^e_{ij}) \to 0$: these edges PASS.
   For perturbed edges with $|W^{e*}_{ij}| \geq \tau$: A2 implies the
   population coefficients differ, so $\mathrm{Var}_e \to c > 0$: these
   FAIL. Edges absent from all environments have estimates $\to 0 <
   \tau$: FAIL.
3. Combining: with probability $\to 1$,
   $\widehat{E}_{\mathrm{inv}} = E^{\tau}_{\mathrm{inv}}$. ∎

**Empirical support.** Exp A: F1 0.945 ± 0.028; Part 1: F1 ≥ 0.889 up to
d = 200 (GPU); phase-2: F1 ≈ 0.93 already at n = 250/env.

---

## Theorem 2 (Non-identifiability of pooled learning — necessity of multi-environment structure)

> **Statement.** There exist two environments with mechanisms
> $\beta_1 \neq \beta_2$ on an edge $(s, y)$ (A1 holding in both) such
> that pooled single-environment learning declares $(s,y)$ an invariance-
> carrying edge with a nonzero coefficient — a false invariance claim —
> while the true invariant graph contains no $(s,y)$ edge. Hence pooled
> (single-graph) estimation cannot recover invariant subgraphs, and
> per-environment structure is necessary.

**Proof sketch.** Let env-1 have $y = \beta_1 s + \varepsilon$ and env-2
$y = \beta_2 s + \varepsilon$, $\beta_1 = 2$, $\beta_2 = 0$. The pooled
least-squares fit converges to the moment-weighted average
$\bar\beta = \lambda_1\beta_1 + \lambda_2\beta_2 = 2\lambda_1 \neq 0$.
Since the pooled learner does not model environments, the edge $(s,y)$
is part of ITS graph with coefficient $\bar\beta$; no statistic from the
pooled fit can detect the heterogeneity. But the true invariant graph
excludes $(s,y)$. ∎

**Empirical support.** NOTEARS-pooled F1 0.851 vs SCIN 0.945 (Exp A);
flat ~0.85 across all sample sizes (phase-2).

---

## Theorem 3 (Conditional OOD risk bound)

> **Statement.** Let $y = \theta^\top x_c + \varepsilon$ with $x_c$ the
> causal parents in the invariant subgraph, and suppose the test
> environment agrees with train on the causal mechanism but is otherwise
> arbitrary in the spurious directions $\beta^{\mathrm{test}}$. If
> SCIN's selection is correct ($\hat c \supseteq c$ per Thm 1), then:
>
> $$R^{\mathrm{test}}(f_{\mathrm{SCIN}}) \le \sigma_\varepsilon^2 +
> o(1) \quad (\text{noise floor})$$
>
> whereas for pooled ERM on all features:
>
> $$R^{\mathrm{test}}(f_{\mathrm{ERM}}) = \sigma_\varepsilon^2 +
> \big\| \beta^{\mathrm{test}} - \hat\beta
> \big\|^2_{\Sigma^{s\,|\,c,\,\mathrm{test}}}$$
>
> — growing without bound in the spurious shift, where
> $\Sigma^{s|c}$ is the conditional second moment of the spurious block.

**Proof sketch.** SCIN's error decomposes as
$y - \hat\theta^\top x_c = (\theta - \hat\theta)^\top x_c +
\varepsilon$: both terms are distribution-agreement-dependent ONLY on
the causal mechanism, which the test environment preserves; the
estimation term vanishes with $n$. ERM predicts with the spurious
coefficients too; its excess risk is the Mahalanobis-type term above,
which the test distribution controls and can make arbitrarily large. ∎

**Empirical support.** Exp B: SCIN == Oracle == noise floor (0.99) while
ERM 10.1; Exp E sign-flip: SCIN 1.06 ≈ Oracle 1.005 vs ERM 1.34. NOTE
the honest flip side (Exp E mild shift): ERM beats SCIN (0.174 vs
0.285) — Thm 3 bounds worst-case robustness, not near-train shifts,
where exploiting the spurious feature still pays. The bound is
conditional on correct selection (A1–A3).

---

## Status

- [x] T2: complete (constructive counterexample).
- [ ] T1: full written proof pending (finite-sample rates via
      concentration of sample covariances per environment).
- [ ] T3: full written proof pending (conditional-risk algebra under
      correct selection + selection-error handling via confidence bounds
      on the invariance statistic).
- [!] A3 is an assumption: the paper must flag it, with the neural
      negative result (RESULTS.md) as evidence of its limits.
