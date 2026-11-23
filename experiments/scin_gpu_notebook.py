# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "scipy", "torch"]
# ///
"""SCIN Phase-3 GPU Experiments — marimo notebook.

Self-contained (no SCIN package needed). Runs on CPU too (slower).
Designed for a molab/Colab GPU session, e.g. RTX PRO 6000 Blackwell 96 GB.

Open with:  marimo edit scin_gpu_notebook.py   (or upload to molab.marimo.io)

Set SMOKE=False in the config cell for the real run (SMOKE=True is a fast
~3-minute pipeline test). The last cell saves gpu_scaleup.json and offers it
for download — please share that file back for analysis.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # SCIN — Phase-3 GPU Experiments
        Sparse Causal Intervention Networks:

        - **Part 1** — linear scale-up, d ∈ {25, 50, 100, 200}
        - **Part 2** — post-nonlinear (PNL) robustness
        - **Part 3** — neural SCIN v0 (MLP encoder + latent SEM) vs ERM-MLP

        Set `SMOKE = False` in the config cell for the real run.
        """
    )
    return ()


@app.cell
def _():
    import json
    import time

    import numpy as np
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp64 REQUIRED: fp32 overflows inside L-BFGS line search once the
    # augmented-Lagrangian rho grows. Negligible cost at d<=200.
    DTYPE = torch.float64
    return DEVICE, DTYPE, json, np, time, torch


@app.cell
def _(DEVICE, DTYPE, mo):
    mo.md(f"**Device:** `{DEVICE}`  |  **dtype:** `{DTYPE}`")
    return ()


@app.cell
def _():
    # ---------------- configuration: edit these ----------------
    SMOKE = True      # True = ~3-min pipeline test; False = real run
    RUN_PART1 = True  # linear scale-up
    RUN_PART2 = True  # PNL robustness
    RUN_PART3 = True  # neural SCIN v0
    HP = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0,
          "tau": 0.2, "var_tol": 0.01}
    return HP, RUN_PART1, RUN_PART2, RUN_PART3, SMOKE


@app.cell
def _(np):
    # ---------------- synthetic SCM generation (inlined) ----------------
    def random_dag(d, n_edges, rng):
        perm = rng.permutation(d)
        A = np.zeros((d, d), dtype=float)
        possible = [(i, j) for i in range(d) for j in range(i + 1, d)]
        idx = rng.choice(len(possible), size=n_edges, replace=False)
        for k in idx:
            i, j = possible[k]
            w = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)
            A[perm[i], perm[j]] = w
        return A

    def simulate_linear_sem(W, n, rng, noise_scale=None):
        d = W.shape[0]
        X = np.zeros((n, d))
        if noise_scale is None:
            noise_scale = np.ones(d)
        Wt = W.copy()
        order = []
        remaining = set(range(d))
        while remaining:
            sources = [v for v in remaining
                       if not any(Wt[u, v] != 0 for u in remaining)]
            if not sources:
                raise ValueError("W is not a DAG")
            order.extend(sorted(sources))
            remaining -= set(sources)
            for v in list(sources):
                Wt[v, :] = 0.0
        for j in order:
            X[:, j] = X @ W[:, j] + rng.normal(0.0, noise_scale[j], size=n)
        return X

    def make_env_data(d, n_edges, n_envs, n_per_env, n_perturbed, seed):
        rng = np.random.default_rng(seed)
        W_base = random_dag(d, n_edges, rng)
        edges = list(zip(*np.nonzero(W_base)))
        perturbed = set()
        if n_perturbed > 0 and len(edges) >= n_perturbed:
            perturbed = set(
                edges[k] for k in
                rng.choice(len(edges), size=n_perturbed, replace=False))
        W_envs = []
        for e in range(n_envs):
            We = W_base.copy()
            if e > 0:
                for (i, j) in perturbed:
                    We[i, j] = W_base[i, j] * rng.choice([0.3, 1.8, -1.0])
            W_envs.append(We)
        Xs = [simulate_linear_sem(We, n_per_env, rng) for We in W_envs]
        W_invariant = np.zeros_like(W_base, dtype=float)
        for (i, j) in edges:
            coeffs = [We[i, j] for We in W_envs]
            if np.std(coeffs) < 1e-9:
                W_invariant[i, j] = W_base[i, j]
        return {"W_base": W_base, "W_envs": W_envs, "W_invariant": W_invariant,
                "W_full": (W_base != 0).astype(float), "Xs": Xs,
                "perturbed_edges": sorted(perturbed)}
    return (make_env_data,)


@app.cell
def _(np):
    # ---------------- metrics ----------------
    def prf(W_true_bin, W_est_bin):
        true = W_true_bin.astype(bool)
        est = W_est_bin.astype(bool)
        tp = int(np.sum(true & est))
        fp = int(np.sum(~true & est))
        fn = int(np.sum(true & ~est))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1

    def shd(W_true_bin, W_est_bin):
        T = W_true_bin.astype(bool)
        E = W_est_bin.astype(bool)
        d = T.shape[0]
        extra = int(np.sum(E & ~T))
        missing_rev = 0
        missing = 0
        counted = np.zeros_like(T)
        for i in range(d):
            for j in range(d):
                if T[i, j] and not E[i, j]:
                    if E[j, i] and not counted[j, i]:
                        missing_rev += 1
                        counted[j, i] = True
                    else:
                        missing += 1
        return extra + missing + missing_rev
    return (prf, shd)


@app.cell
def _(torch):
    def h_poly(Wm):
        """NOTEARS polynomial acyclicity constraint (overflow-safe)."""
        dim_ = Wm.shape[0]
        M_ = torch.eye(dim_, device=Wm.device, dtype=Wm.dtype) + Wm * Wm / dim_
        return torch.linalg.matrix_power(M_, dim_).diagonal().sum() - dim_
    return (h_poly,)


@app.cell
def _(DTYPE, DEVICE, HP, h_poly, np, torch):
    # ------------- fitters (L-BFGS inner solver — REQUIRED) -------------
    def notears_gpu(X_dev, lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                    scales_dev=None, h_tol=1e-6, rho_max=1e12,
                    max_iter=200, seed=0):
        torch.manual_seed(seed)
        n_, d_ = X_dev.shape
        mask_ = 1.0 - torch.eye(d_, device=DEVICE, dtype=DTYPE)
        sc2_ = (torch.ones(d_, device=DEVICE, dtype=DTYPE)
                if scales_dev is None else scales_dev ** 2)
        W_ = torch.zeros(d_, d_, device=DEVICE, requires_grad=True)
        rho_, alpha_, h_prev_ = 1.0, 0.0, float("inf")
        for _outer_ in range(100):
            opt_ = torch.optim.LBFGS([W_], max_iter=max_iter,
                                     tolerance_grad=1e-7, tolerance_change=1e-11,
                                     line_search_fn="strong_wolfe")

            def closure_():
                opt_.zero_grad()
                Wm_ = W_ * mask_
                loss_ = 0.5 / n_ * torch.sum((X_dev - X_dev @ Wm_) ** 2)
                if lambda_ev != 0.0:
                    r_ = X_dev - X_dev @ Wm_
                    logv_ = torch.log(r_.var(dim=0, unbiased=False) * sc2_
                                      + 1e-12)
                    loss_ = loss_ + lambda_ev * ((logv_ - logv_.mean()) ** 2).sum()
                hg_ = h_poly(Wm_)
                obj_ = (loss_ + lambda1 * W_.abs().sum() + 0.5 * rho_ * hg_ ** 2
                        + alpha_ * hg_)
                obj_.backward()
                return obj_

            opt_.step(closure_)
            with torch.no_grad():
                hg_ = float(h_poly(W_ * mask_))
            if hg_ < h_tol or rho_ >= rho_max:
                break
            if hg_ > 0.25 * h_prev_:
                rho_ *= 10
            else:
                alpha_ += rho_ * hg_
            h_prev_ = hg_
        return (W_ * mask_).detach().cpu().numpy().astype(np.float64)

    def scin_gpu(Xs_dev, lambda1=HP["lambda1"], lambda3=HP["lambda3"],
                 lambda_ev=HP["lambda_ev"], scales_dev=None, h_tol=1e-6,
                 rho_max=1e12, max_iter=200, W_init=None, seed=0):
        torch.manual_seed(seed)
        d_ = Xs_dev[0].shape[1]
        E_ = len(Xs_dev)
        ns_ = [X.shape[0] for X in Xs_dev]
        mask_ = 1.0 - torch.eye(d_, device=DEVICE, dtype=DTYPE)
        sc2_ = (torch.ones(d_, device=DEVICE, dtype=DTYPE)
                if scales_dev is None else scales_dev ** 2)
        W_ = ((torch.zeros(d_, d_, device=DEVICE) if W_init is None
               else W_init.clone()).requires_grad_(True))
        Deltas_ = [torch.zeros(d_, d_, device=DEVICE, requires_grad=True)
                   for _ in range(E_)]
        rho_, alpha_, h_prev_ = 1.0, 0.0, float("inf")
        for _outer_ in range(100):
            opt_ = torch.optim.LBFGS([W_] + Deltas_, max_iter=max_iter,
                                     tolerance_grad=1e-7,
                                     tolerance_change=1e-11,
                                     line_search_fn="strong_wolfe")

            def closure_():
                opt_.zero_grad()
                Wm_ = W_ * mask_
                Wse_ = [(W_ + D) * mask_ for D in Deltas_]
                recon_ = sum(0.5 / ns_[e] *
                             torch.sum((Xs_dev[e] - Xs_dev[e] @ Wse_[e]) ** 2)
                             for e in range(E_))
                l1_ = W_.abs().sum()
                ridge_ = lambda3 * sum((D ** 2).sum() for D in Deltas_)
                ev_ = torch.zeros((), device=DEVICE, dtype=DTYPE)
                if lambda_ev != 0.0:
                    for e in range(E_):
                        r_ = Xs_dev[e] - Xs_dev[e] @ Wse_[e]
                        logv_ = torch.log(r_.var(dim=0, unbiased=False) * sc2_
                                          + 1e-12)
                        ev_ = ev_ + ((logv_ - logv_.mean()) ** 2).sum()
                hs_ = [h_poly(We) for We in Wse_]
                obj_ = (recon_ + lambda1 * l1_ + ridge_ + lambda_ev * ev_
                        + sum(0.5 * rho_ * he_ ** 2 + alpha_ * he_
                              for he_ in hs_))
                obj_.backward()
                return obj_

            opt_.step(closure_)
            with torch.no_grad():
                hg_ = float(sum(h_poly((W_ + D) * mask_) for D in Deltas_))
            if hg_ < h_tol or rho_ >= rho_max:
                break
            if hg_ > 0.25 * h_prev_:
                rho_ *= 10
            else:
                alpha_ += rho_ * hg_
            h_prev_ = hg_
        with torch.no_grad():
            Wse_ = [(W_ + D) * mask_ for D in Deltas_]
            return (W_.detach().cpu().numpy().astype(np.float64),
                    [We.cpu().numpy().astype(np.float64) for We in Wse_])
    return (notears_gpu, scin_gpu)


@app.cell
def _(DTYPE, DEVICE, np, torch):
    def make_env_gpu(d1, n_per1, envs1, npert1, seed1, make_env_data):
        data_ = make_env_data(d=d1, n_edges=int(1.2 * d1), n_envs=envs1,
                              n_per_env=n_per1, n_perturbed=npert1, seed=seed1)
        Xs_dev_ = [torch.as_tensor(X, dtype=DTYPE, device=DEVICE)
                   for X in data_["Xs"]]
        sd1_ = np.vstack(data_["Xs"]).std(0) + 1e-12
        sd_dev_ = torch.as_tensor(sd1_, dtype=DTYPE, device=DEVICE)
        return data_, Xs_dev_, sd1_, sd_dev_
    return (make_env_gpu,)


@app.cell
def _(json, time):
    RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    JSON_PATH = "gpu_scaleup.json"
    return JSON_PATH, RESULTS


@app.cell
def _(RUN_PART1, SMOKE, mo):
    mo.stop(not RUN_PART1)
    mo.md(f"## Part 1 — linear scale-up{' (SMOKE)' if SMOKE else ''}")
    return ()


@app.cell
def _(DEVICE, DTYPE, HP, RUN_PART1, SMOKE, make_env_data, make_env_gpu,
      notears_gpu, np, prf, scin_gpu, shd, time, torch):
    def run_part1():
        rows_ = []
        d_list_ = [20] if SMOKE else [25, 50, 100, 200]
        n_per_ = 400 if SMOKE else 2000
        max_it_ = 80 if SMOKE else 150
        for d1_ in d_list_:
            data_, Xs_dev_, sd_, sd_dev_ = make_env_gpu(
                d1_, n_per_, 2, max(3, d1_ // 5), seed1=0,
                make_env_data=make_env_data)
            t0_ = time.time()
            W_pool_ = notears_gpu(torch.vstack(Xs_dev_), scales_dev=sd_dev_,
                                  seed=0, max_iter=max_it_)
            t_warm_ = time.time() - t0_
            t0_ = time.time()
            _, Ws_out_ = scin_gpu(Xs_dev_, scales_dev=sd_dev_, seed=0,
                                  max_iter=max_it_,
                                  W_init=torch.as_tensor(W_pool_,
                                                         dtype=DTYPE,
                                                         device=DEVICE))
            t_scin_ = time.time() - t0_
            st_ = np.stack(Ws_out_)
            est_ = ((st_.var(0) < HP["var_tol"]) &
                    (np.abs(st_).mean(0) > HP["tau"])).astype(float)
            prec_, rec_, f1_ = prf(data_["W_invariant"], est_)
            rows_.append({
                "d": d1_, "n_per_env": n_per_, "device": str(DEVICE),
                "warmstart_s": round(t_warm_, 1), "scin_s": round(t_scin_, 1),
                "precision": round(prec_, 3), "recall": round(rec_, 3),
                "f1": round(f1_, 3), "shd": shd(data_["W_invariant"], est_)})
            print(f"d={d1_:>4}: F1={f1_:.3f} SHD={rows_[-1]['shd']} "
                  f"warm={t_warm_:.1f}s scin={t_scin_:.1f}s")
        return rows_

    part1_rows = run_part1() if RUN_PART1 else None
    return (part1_rows,)


@app.cell
def _(RUN_PART2, SMOKE, mo):
    mo.stop(not RUN_PART2)
    mo.md(f"## Part 2 — post-nonlinear (PNL) robustness{' (SMOKE)' if SMOKE else ''}")
    return ()


@app.cell
def _(DTYPE, DEVICE, HP, RUN_PART2, SMOKE, make_env_data, notears_gpu, np,
      prf, scin_gpu, torch):
    def run_part2():
        rows2_ = []

        def distort2(X, strength_):
            return X + strength_ * 0.5 * np.tanh(X / 0.5)

        strengths_ = [0.0, 0.5] if SMOKE else [0.0, 0.25, 0.5, 1.0]
        for strength_ in strengths_:
            f1s_ = []
            for seed2_ in (range(2) if SMOKE else range(3)):
                data_ = make_env_data(d=10, n_edges=10, n_envs=2,
                                      n_per_env=300 if SMOKE else 1000,
                                      n_perturbed=3, seed=seed2_)
                Xs2_ = [distort2(X, strength_) for X in data_["Xs"]]
                sd2_ = np.vstack(Xs2_).std(0) + 1e-12
                Xs_std2_ = [(X - np.vstack(Xs2_).mean(0)) / sd2_ for X in Xs2_]
                Xp_dev_ = torch.as_tensor(np.vstack(Xs_std2_),
                                          dtype=DTYPE, device=DEVICE)
                sd_dev_ = torch.as_tensor(sd2_, dtype=DTYPE, device=DEVICE)
                W_pool_ = notears_gpu(Xp_dev_, scales_dev=sd_dev_,
                                      seed=seed2_,
                                      max_iter=60 if SMOKE else 150)
                _, Ws_ = scin_gpu([torch.as_tensor(X, dtype=DTYPE,
                                                   device=DEVICE)
                                   for X in Xs_std2_],
                                  scales_dev=sd_dev_, seed=seed2_,
                                  max_iter=80 if SMOKE else 200,
                                  W_init=torch.as_tensor(W_pool_,
                                                         dtype=DTYPE,
                                                         device=DEVICE))
                st2_ = np.stack(Ws_)
                est2_ = ((st2_.var(0) < HP["var_tol"]) &
                         (np.abs(st2_).mean(0) > HP["tau"])).astype(float)
                f1s_.append(prf(data_["W_invariant"], est2_)[2])
            rows2_.append({"strength": strength_,
                           "f1_mean": round(float(np.mean(f1s_)), 3),
                           "f1_std": round(float(np.std(f1s_)), 3)})
            print(f"PNL strength={strength_}: F1 {rows2_[-1]['f1_mean']}"
                  f"±{rows2_[-1]['f1_std']}")
        return rows2_

    part2_rows = run_part2() if RUN_PART2 else None
    return (part2_rows,)


@app.cell
def _(RUN_PART3, SMOKE, mo):
    mo.stop(not RUN_PART3)
    mo.md(f"## Part 3 — neural SCIN v0 vs ERM-MLP{' (SMOKE)' if SMOKE else ''}")
    return ()


@app.cell
def _(DEVICE, RUN_PART3, SMOKE, h_poly, np, torch):
    def run_part3():
        rng3_ = np.random.default_rng(0)
        betas_tr_, betas_te_ = (2.0, 1.0), (0.0, -2.0)
        n3_ = 800 if SMOKE else 4000
        d_lat_ = 16

        def sample3(beta_, n_):
            X_ = rng3_.normal(size=(n_, 6))  # x0,x1,x2,s,z1,z2
            y_ = (1.5 * np.tanh(X_[:, 1]) - 1.0 * (X_[:, 2] ** 2 - 1.0)
                  + 0.5 * rng3_.normal(size=n_))
            X_[:, 3] = beta_ * y_ + rng3_.normal(size=n_)  # spurious (env-dep)
            # z1, z2 (cols 4,5) stay noise
            return X_.astype(np.float32), y_.astype(np.float32)

        tr3_ = [sample3(b, n3_) for b in betas_tr_]
        te3_ = [sample3(b, n3_) for b in betas_te_]

        def mse3(y_, yh_):
            return float(np.mean((y_ - np.asarray(yh_)) ** 2))

        # ---- ERM-MLP baseline
        torch.manual_seed(0)
        erm_ = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, 1)).to(DEVICE)
        opt_erm_ = torch.optim.Adam(erm_.parameters(), lr=1e-3)
        Xtr_ = torch.as_tensor(np.vstack([t[0] for t in tr3_]), device=DEVICE)
        ytr_ = torch.as_tensor(np.hstack([t[1] for t in tr3_]), device=DEVICE)
        for epoch_ in range(400 if SMOKE else 3000):
            opt_erm_.zero_grad()
            ((erm_(Xtr_).squeeze() - ytr_) ** 2).mean().backward()
            opt_erm_.step()
        erm_res_ = {}
        with torch.no_grad():
            for tag_, env_ in [("ood0", te3_[0]), ("ood_neg", te3_[1])]:
                Xe_ = torch.as_tensor(env_[0], device=DEVICE)
                erm_res_[tag_] = round(mse3(env_[1],
                                            erm_(Xe_).squeeze().cpu().numpy()), 4)

        # ---- Neural SCIN v0: encoder z=MLP(x); linear SEM over [z, y] with
        # per-env deltas + ev penalty; select latent ancestors of y; ridge.
        torch.manual_seed(1)
        enc_ = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU(),
                                   torch.nn.Linear(32, d_lat_)).to(DEVICE)
        lat_ = [torch.as_tensor(t[0], device=DEVICE) for t in tr3_]
        y_dev_ = [torch.as_tensor(t[1], device=DEVICE) for t in tr3_]
        dtot_ = d_lat_ + 1
        mask3_ = 1.0 - torch.eye(dtot_, device=DEVICE)
        Ws3_np_ = None
        for phase_ in range(2 if SMOKE else 3):
            with torch.no_grad():
                Zs_raw_ = [enc_(x) for x in lat_]
                Zall_ = torch.vstack(Zs_raw_)
                mu_, sdm_ = Zall_.mean(0), Zall_.std(0) + 1e-6
            Zs_ = [(z - mu_) / sdm_ for z in Zs_raw_]
            Zsy_ = [torch.cat([z, y.unsqueeze(1)], 1)
                    for z, y in zip(Zs_, y_dev_)]
            sc23_ = torch.cat([torch.ones(dtot_ - 1, device=DEVICE),
                               (y_dev_[0].var() + 1e-6).reshape(1)])
            Wm_ = torch.zeros(dtot_, dtot_, device=DEVICE, requires_grad=True)
            Dl_ = [torch.zeros(dtot_, dtot_, device=DEVICE, requires_grad=True)
                   for _ in range(2)]
            al_ = torch.optim.Adam([Wm_] + Dl_, lr=0.02)
            rho_, alph_, hp_ = 1.0, 0.0, float("inf")
            for outer_ in range(10 if SMOKE else 30):
                for _inner_ in range(40 if SMOKE else 150):
                    al_.zero_grad()
                    Wse_ = [(Wm_ + D) * mask3_ for D in Dl_]
                    rec_ = sum(0.5 / len(Z) * torch.sum((Z - Z @ Wse_[e]) ** 2)
                               for e, Z in enumerate(Zsy_))
                    ridge_ = 0.1 * sum((D ** 2).sum() for D in Dl_)
                    l1_ = 0.1 * Wm_.abs().sum()
                    ev_ = torch.zeros((), device=DEVICE)
                    for e, Z in enumerate(Zsy_):
                        r_ = Z - Z @ Wse_[e]
                        logv_ = torch.log(r_.var(dim=0, unbiased=False) * sc23_
                                          + 1e-12)
                        ev_ = ev_ + ((logv_ - logv_.mean()) ** 2).sum()
                    hs_ = [h_poly(We) for We in Wse_]
                    obj_ = (rec_ + l1_ + ridge_ + 1.0 * ev_
                            + sum(0.5 * rho_ * he_ ** 2 + alph_ * he_
                                  for he_ in hs_))
                    obj_.backward()
                    al_.step()
                with torch.no_grad():
                    hg_ = float(sum(h_poly((Wm_ + D) * mask3_) for D in Dl_))
                if hg_ < 1e-4:
                    break
                if hg_ > 0.25 * hp_:
                    rho_ *= 10
                else:
                    alph_ += rho_ * hg_
                hp_ = hg_
            with torch.no_grad():
                Ws3_np_ = [((Wm_ + D) * mask3_).cpu().numpy() for D in Dl_]

        st3_ = np.stack(Ws3_np_)
        adj3_ = ((st3_.var(0) < 0.05) &
                 (np.abs(st3_).mean(0) > 0.1)).astype(float)
        anc_, frontier3_ = set(), [dtot_ - 1]
        while frontier3_:
            j3_ = frontier3_.pop()
            for i3_ in range(dtot_ - 1):
                if adj3_[i3_, j3_] and i3_ not in anc_:
                    anc_.add(i3_)
                    frontier3_.append(i3_)
        sel3_ = sorted(anc_)
        if sel3_:
            with torch.no_grad():
                Ztr_sel_ = [((enc_(x) - mu_) / sdm_)[:, sel3_]
                            for x in lat_]
            Znp_ = torch.vstack(Ztr_sel_).cpu().numpy()
            ynp_ = np.hstack([t[1] for t in tr3_])
            Xb_ = np.hstack([Znp_, np.ones((len(Znp_), 1))])
            A_ = Xb_.T @ Xb_ + 1e-2 * np.eye(Xb_.shape[1])
            A_[-1, -1] -= 1e-2
            wb_ = np.linalg.solve(A_, Xb_.T @ ynp_)
            w_, b_ = wb_[:-1], wb_[-1]
            scin_res_ = {}
            for tag_, env_ in [("ood0", te3_[0]), ("ood_neg", te3_[1])]:
                with torch.no_grad():
                    Z3_ = ((((enc_(torch.as_tensor(env_[0], device=DEVICE))
                              - mu_) / sdm_))[:, sel3_])
                    yh3_ = Z3_.cpu().numpy() @ w_ + b_
                scin_res_[tag_] = round(mse3(env_[1], yh3_), 4)
        else:
            scin_res_ = {"ood0": float("nan"), "ood_neg": float("nan"),
                         "note": "no latents selected"}
        return {"erm_ood_mse": erm_res_, "scin_v0_mse": scin_res_,
                "n_latents_selected": len(sel3_), "d_latent": d_lat_}

    part3_res = run_part3() if RUN_PART3 else None
    return (part3_res,)


@app.cell
def _(DEVICE, JSON_PATH, RESULTS, json, mo, np, part1_rows, part2_rows,
      part3_res):
    for key_, val_ in [("linear_scaleup", part1_rows), ("pnl", part2_rows),
                       ("neural_v0", part3_res)]:
        if val_ is not None:
            RESULTS[key_] = val_
    RESULTS["device"] = str(DEVICE)
    payload_ = {k: v for k, v in RESULTS.items() if v is not None}
    mo.md("Saved `gpu_scaleup.json` — download it and share it back.")
    mo.download(data=json.dumps(payload_, indent=2, default=str),
                filename="gpu_scaleup.json", label="Download gpu_scaleup.json")
    return (payload_,)


if __name__ == "__main__":
    app.run()
