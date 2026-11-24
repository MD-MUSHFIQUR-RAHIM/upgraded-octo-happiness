#@title 🧠 SCIN max-effort GPU package: E-scaling (real data) + 10-seed chambers + d=300/400 {display-mode:"form"}
# Paste this single cell into Colab (GPU runtime). No pip installs needed.
# Full run ≈ 60-80 min (fp64 is REQUIRED — do NOT switch to fp32/TF32).
# Output: prints one JSON at the end — paste it back to chat.

SMOKE = False   # ← False = full (delivered state). Set True for a ~3-min test.

import io, json, time, zipfile
import numpy as np
import pandas as pd
import requests
import torch

torch.set_default_dtype(torch.float64)          # fp64 REQUIRED (L-BFGS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
print("Device:", DEVICE)

# ============================ ground truth (lt/standard) =====================
GT_EDGES = [
    ("red","current"),("green","current"),("blue","current"),
    ("osr_c","current"),("v_c","current"),
    ("red","ir_1"),("red","vis_1"),("red","ir_2"),("red","vis_2"),
    ("red","ir_3"),("red","vis_3"),
    ("green","ir_1"),("green","vis_1"),("green","ir_2"),("green","vis_2"),
    ("green","ir_3"),("green","vis_3"),
    ("blue","ir_1"),("blue","vis_1"),("blue","ir_2"),("blue","vis_2"),
    ("blue","ir_3"),("blue","vis_3"),
    ("pol_1","angle_1"),("pol_1","ir_3"),("pol_1","vis_3"),
    ("pol_2","angle_2"),("pol_2","ir_3"),("pol_2","vis_3"),
    ("osr_angle_1","angle_1"),("osr_angle_2","angle_2"),
    ("v_angle_1","angle_1"),("v_angle_2","angle_2"),
    ("l_11","ir_1"),("l_11","vis_1"),("l_12","ir_1"),("l_12","vis_1"),
    ("l_21","ir_2"),("l_21","vis_2"),("l_22","ir_2"),("l_22","vis_2"),
    ("l_31","ir_3"),("l_31","vis_3"),("l_32","ir_3"),("l_32","vis_3"),
    ("diode_ir_1","ir_1"),("diode_vis_1","vis_1"),
    ("diode_ir_2","ir_2"),("diode_vis_2","vis_2"),
    ("diode_ir_3","ir_3"),("diode_vis_3","vis_3"),
    ("t_ir_1","ir_1"),("t_vis_1","vis_1"),
    ("t_ir_2","ir_2"),("t_vis_2","vis_2"),
    ("t_ir_3","ir_3"),("t_vis_3","vis_3"),
]

# ============================ metrics ========================================
def prf_(Wt, We):
    t, e = Wt.astype(bool), We.astype(bool)
    tp = int(np.sum(t & e)); fp = int(np.sum(~t & e)); fn = int(np.sum(t & ~e))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def shd_(Wt, We):
    T, E = Wt.astype(bool), We.astype(bool)
    extra = int(np.sum(E & ~T)); miss = 0; rev = 0
    cnt = np.zeros_like(T)
    d = T.shape[0]
    for i in range(d):
        for j in range(d):
            if T[i, j] and not E[i, j]:
                if E[j, i] and not cnt[j, i]:
                    rev += 1; cnt[j, i] = True
                else:
                    miss += 1
    return extra + miss + rev

def skeleton_(B):
    S = (np.abs(B) > 0).astype(float)
    return ((S + S.T) > 0).astype(float)

def scin_select_masked_(st_, envs_, tau_, var_tol_):
    """Masked invariance selection (see colab_expg_cell.py)."""
    d_ = st_.shape[1]
    env_sd_ = np.stack([e.std(0) for e in envs_])
    vary_ = env_sd_ > 0.1
    adj_ = np.zeros((d_, d_))
    for i in range(d_):
        viv = vary_[:, i]
        if viv.sum() < 3:
            continue
        vi_ = st_[:, i, :][viv].var(axis=0)
        mi_ = np.abs(st_[:, i, :])[viv].mean(axis=0)
        adj_[i, :] = ((vi_ < var_tol_) & (mi_ > tau_)).astype(float)
    return adj_

# ============================ fitters (fp64 + L-BFGS REQUIRED) ===============
def h_poly_(Wm):
    d_ = Wm.shape[0]
    M_ = torch.eye(d_, device=Wm.device, dtype=Wm.dtype) + Wm * Wm / d_
    return torch.linalg.matrix_power(M_, d_).diagonal().sum() - d_

def notears_gpu_(X_dev, lambda1=0.1, lambda_ev=1.0, scales_dev=None,
                 h_tol=1e-6, rho_max=1e12, max_iter=200, seed=0):
    torch.manual_seed(seed)
    n_, d_ = X_dev.shape
    mask_ = 1.0 - torch.eye(d_, device=DEVICE, dtype=DTYPE)
    sc2_ = (torch.ones(d_, device=DEVICE, dtype=DTYPE)
            if scales_dev is None else scales_dev ** 2)
    W_ = torch.zeros(d_, d_, device=DEVICE, requires_grad=True)
    rho_, alpha_, h_prev_ = 1.0, 0.0, float("inf")
    for _o_ in range(100):
        opt_ = torch.optim.LBFGS([W_], max_iter=max_iter, tolerance_grad=1e-7,
                                 tolerance_change=1e-11,
                                 line_search_fn="strong_wolfe")
        def closure_():
            opt_.zero_grad()
            Wm_ = W_ * mask_
            loss_ = 0.5 / n_ * torch.sum((X_dev - X_dev @ Wm_) ** 2)
            if lambda_ev != 0.0:
                r_ = X_dev - X_dev @ Wm_
                logv_ = torch.log(r_.var(dim=0, unbiased=False) * sc2_ + 1e-12)
                loss_ = loss_ + lambda_ev * ((logv_ - logv_.mean()) ** 2).sum()
            hg_ = h_poly_(Wm_)
            obj_ = (loss_ + lambda1 * W_.abs().sum()
                    + 0.5 * rho_ * hg_ ** 2 + alpha_ * hg_)
            obj_.backward()
            return obj_
        opt_.step(closure_)
        with torch.no_grad():
            hg_ = float(h_poly_(W_ * mask_))
        if hg_ < h_tol or rho_ >= rho_max:
            break
        if hg_ > 0.25 * h_prev_:
            rho_ *= 10
        else:
            alpha_ += rho_ * hg_
        h_prev_ = hg_
    return (W_ * mask_).detach().cpu().numpy().astype(np.float64)

def scin_gpu_(Xs_dev, lambda1=0.1, lambda3=0.1, lambda_ev=1.0,
              scales_dev=None, h_tol=1e-6, rho_max=1e12, max_iter=200,
              W_init=None, seed=0):
    torch.manual_seed(seed)
    d_ = Xs_dev[0].shape[1]; E_ = len(Xs_dev)
    ns_ = [X.shape[0] for X in Xs_dev]
    mask_ = 1.0 - torch.eye(d_, device=DEVICE, dtype=DTYPE)
    sc2_ = (torch.ones(d_, device=DEVICE, dtype=DTYPE)
            if scales_dev is None else scales_dev ** 2)
    W_ = ((torch.zeros(d_, d_, device=DEVICE, dtype=DTYPE) if W_init is None
           else W_init.clone()).requires_grad_(True))
    Deltas_ = [torch.zeros(d_, d_, device=DEVICE, dtype=DTYPE,
                           requires_grad=True) for _ in range(E_)]
    rho_, alpha_, h_prev_ = 1.0, 0.0, float("inf")
    for _o_ in range(100):
        opt_ = torch.optim.LBFGS([W_] + Deltas_, max_iter=max_iter,
                                 tolerance_grad=1e-7, tolerance_change=1e-11,
                                 line_search_fn="strong_wolfe")
        def closure_():
            opt_.zero_grad()
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
            hs_ = [h_poly_(We) for We in Wse_]
            obj_ = (recon_ + lambda1 * l1_ + ridge_ + lambda_ev * ev_
                    + sum(0.5 * rho_ * he_ ** 2 + alpha_ * he_ for he_ in hs_))
            obj_.backward()
            return obj_
        opt_.step(closure_)
        with torch.no_grad():
            hg_ = float(sum(h_poly_((W_ + D) * mask_) for D in Deltas_))
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

# ============================ synthetic GT (replica of scin.scm) =============
def random_dag_(d_, n_edges_, rng):
    perm_ = rng.permutation(d_)
    A_ = np.zeros((d_, d_))
    poss_ = [(i, j) for i in range(d_) for j in range(i + 1, d_)]
    idx_ = rng.choice(len(poss_), size=n_edges_, replace=False)
    for k in idx_:
        i, j = poss_[k]
        A_[perm_[i], perm_[j]] = (rng.choice([-1.0, 1.0]) *
                                  rng.uniform(0.5, 1.5))
    return A_

def sim_sem_(W, n, rng):
    # topological-order simulation (random_dag_ places edges in perm-space,
    # so the column index order is NOT topological — sort like scin.scm does)
    d_ = W.shape[0]
    X = np.zeros((n, d_))
    Wt = W.copy()
    order, remaining = [], set(range(d_))
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
        X[:, j] = X @ W[:, j] + rng.normal(size=n)
    return X

def make_env_(d_, n_edges_, n_envs_, n_per_, n_pert_, seed):
    rng = np.random.default_rng(seed)
    W_base = random_dag_(d_, n_edges_, rng)
    edges_ = list(zip(*np.nonzero(W_base)))
    pert_ = set(edges_[k] for k in
                rng.choice(len(edges_), size=n_pert_, replace=False))
    W_envs_ = []
    for e in range(n_envs_):
        We = W_base.copy()
        if e > 0:
            for (i, j) in pert_:
                We[i, j] = W_base[i, j] * rng.choice([0.3, 1.8, -1.0])
        W_envs_.append(We)
    Xs = [sim_sem_(We, n_per_, rng) for We in W_envs_]
    W_inv = W_base.copy()
    for (i, j) in pert_:
        W_inv[i, j] = 0.0
    return W_inv, Xs

# ============================ data download ==================================
URL_ = ("https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/"
        "lt_interventions_standard_v1.zip")
ZDATA_ = zipfile.ZipFile(io.BytesIO(requests.get(URL_, timeout=600).content))
print("dataset loaded:",
      len([n for n in ZDATA_.namelist() if n.endswith('.csv')]),
      "experiments")

VARS18 = ["red", "green", "blue", "current", "pol_1", "pol_2",
          "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3",
          "l_11", "l_12", "l_21", "l_22", "l_31", "l_32"]
MAIN12_ = ["reference", "red_mid", "green_mid", "blue_mid",
           "l_11_mid", "l_12_mid", "l_21_mid", "l_22_mid",
           "l_31_mid", "l_32_mid", "pol_1_mid", "pol_2_mid"]
ENV_SETS = {
    4:  ["reference", "red_mid", "green_mid", "blue_mid"],
    8:  ["reference", "red_mid", "green_mid", "blue_mid",
         "l_11_mid", "l_21_mid", "l_31_mid", "pol_1_mid"],
    12: MAIN12_,
    17: ["reference", "red_mid", "red_strong", "green_mid", "green_strong",
         "blue_mid", "blue_strong",
         "l_11_mid", "l_12_mid", "l_21_mid", "l_22_mid",
         "l_31_mid", "l_32_mid", "pol_1_mid", "pol_1_strong",
         "pol_2_mid", "pol_2_strong"],
} if not SMOKE else {
    3: ["reference", "red_mid", "green_mid"],
    5: ["reference", "red_mid", "green_mid", "blue_mid", "l_11_mid"],
}
LADDER_ = sorted(ENV_SETS.keys())

def gt_matrix_(vars_, edges_):
    W = np.zeros((len(vars_), len(vars_)))
    ii = {n: i for i, n in enumerate(vars_)}
    for u, v in edges_:
        if u in ii and v in ii:
            W[ii[u], ii[v]] = 1.0
    return W

def load_envs_(env_names, vars_, n_per, drop_first, seed):
    raw = []
    for k, en in enumerate(env_names):
        with ZDATA_.open(f"lt_interventions_standard_v1/uniform_{en}.csv") as f_:
            df = pd.read_csv(f_)
        X = df[vars_].to_numpy(dtype=np.float64)[drop_first:]
        if SMOKE:
            X = X[:n_per]
        else:
            rng = np.random.default_rng(100000 * seed + 100 * k + len(en))
            X = X[rng.permutation(len(X))[:n_per]]
        raw.append(X)
    pooled = np.vstack(raw)
    mu_, sd_ = pooled.mean(0), pooled.std(0) + 1e-12
    return [(X - mu_) / sd_ for X in raw], sd_

HP_ = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0,
       "tau": 0.1, "var_tol": 0.02}
TAU_SWEEP_ = [0.05, 0.1] if SMOKE else [0.02, 0.05, 0.1, 0.15, 0.2]
out_ = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE), "smoke": SMOKE, "parts": {}}

# =============================================================================
# PART 1 — environment-count scaling on Causal Chambers (18 nodes, real data)
# Question: does SCIN's recovery improve as E grows (more heterogeneity signal)?
# =============================================================================
try:
    t0_ = time.time()
    seeds1_ = [0] if SMOKE else [0, 1, 2]
    W_gt_ = gt_matrix_(VARS18, GT_EDGES)
    rows1_ = []
    for E_n in LADDER_:
        env_names_ = ENV_SETS[E_n]
        for seed in seeds1_:
            envs_, sd_ = load_envs_(env_names_, VARS18, 500, 60, seed)
            sd_dev_ = torch.as_tensor(sd_, device=DEVICE, dtype=DTYPE)
            Xp_ = torch.as_tensor(np.vstack(envs_), device=DEVICE, dtype=DTYPE)
            W_p_ = notears_gpu_(Xp_, lambda1=HP_["lambda1"],
                                lambda_ev=HP_["lambda_ev"], scales_dev=sd_dev_,
                                seed=seed)
            _, Ws_ = scin_gpu_([torch.as_tensor(Xe, device=DEVICE, dtype=DTYPE)
                                for Xe in envs_],
                               lambda1=HP_["lambda1"], lambda3=HP_["lambda3"],
                               lambda_ev=HP_["lambda_ev"], scales_dev=sd_dev_,
                               seed=seed, W_init=None)
            st_ = np.stack(Ws_)
            for tau_ in TAU_SWEEP_:
                b_p_ = (np.abs(W_p_) > tau_).astype(float)
                p_, r_, f_ = prf_(W_gt_, b_p_)
                sk_ = prf_(skeleton_(W_gt_), skeleton_(b_p_))[2]
                rows1_.append({"E": E_n, "m": "NOTEARS-pooled", "tau": tau_,
                               "precision": p_, "recall": r_, "f1": f_,
                               "skel_f1": sk_, "shd": shd_(W_gt_, b_p_)})
                b_i_ = np.ones_like(np.abs(W_p_))
                for We in Ws_:
                    b_i_ = b_i_ * (np.abs(We) > tau_).astype(float)
                p_, r_, f_ = prf_(W_gt_, b_i_)
                sk_ = prf_(skeleton_(W_gt_), skeleton_(b_i_))[2]
                rows1_.append({"E": E_n, "m": "NOTEARS-env-intersection",
                               "tau": tau_, "precision": p_, "recall": r_,
                               "f1": f_, "skel_f1": sk_,
                               "shd": shd_(W_gt_, b_i_)})
                for vt_ in [HP_["var_tol"]] + ([0.05] if not SMOKE else []):
                    b_m_ = scin_select_masked_(st_, envs_, tau_, vt_)
                    lbl_ = ("SCIN (ours, masked)"
                            if vt_ == HP_["var_tol"] else f"SCIN masked vt={vt_}")
                    p_, r_, f_ = prf_(W_gt_, b_m_)
                    sk_ = prf_(skeleton_(W_gt_), skeleton_(b_m_))[2]
                    rows1_.append({"E": E_n, "m": lbl_, "tau": tau_,
                                   "precision": p_, "recall": r_, "f1": f_,
                                   "skel_f1": sk_, "shd": shd_(W_gt_, b_m_)})
        print(f"[E-scaling] E={E_n} done [{time.time()-t0_:.0f}s]")
    summ1_ = {}
    for m_ in sorted(set(r_["m"] for r_ in rows1_)):
        summ1_[m_] = {}
        for E_n in LADDER_:
            rr_ = [r_ for r_ in rows1_ if r_["m"] == m_ and r_["E"] == E_n]
            best_ = max(rr_, key=lambda r_: r_["f1"])
            summ1_[m_][str(E_n)] = {
                "best_tau": best_["tau"], "f1": round(best_["f1"], 3),
                "precision": round(best_["precision"], 3),
                "recall": round(best_["recall"], 3),
                "shd": round(float(best_["shd"]), 1)}
    sel_counts_ = {}
    if not SMOKE:
        for E_n in LADDER_:
            envs_e_, _ = load_envs_(ENV_SETS[E_n], VARS18, 500, 60, 0)
            sd_e_ = np.stack([e.std(0) for e in envs_e_])
            # parents with >=3 varying envs are pollable by SCIN
            sel_counts_[str(E_n)] = int(
                ((sd_e_ > 0.1).sum(0) >= 3).sum())
    out_["parts"]["part1_E_scaling"] = {
        "n_seeds": len(seeds1_), "summary_by_E": summ1_,
        "n_parents_with_3plus_varying_envs": sel_counts_,
        "time_s": round(time.time() - t0_, 1)}
except Exception as ex:
    import traceback
    out_["parts"]["part1_E_scaling"] = {"error": repr(ex),
                                        "trace": traceback.format_exc()[-1500:]}

# =============================================================================
# PART 2 — 10-seed Causal Chambers main protocol (12 envs, 18 nodes)
# =============================================================================
try:
    t0_ = time.time()
    seeds2_ = [0] if SMOKE else list(range(10))
    env_names_ = MAIN12_
    W_gt_ = gt_matrix_(VARS18, GT_EDGES)
    rows2_ = []
    for seed in seeds2_:
        envs_, sd_ = load_envs_(env_names_, VARS18, 500, 60, seed)
        sd_dev_ = torch.as_tensor(sd_, device=DEVICE, dtype=DTYPE)
        Xp_ = torch.as_tensor(np.vstack(envs_), device=DEVICE, dtype=DTYPE)
        W_p_ = notears_gpu_(Xp_, lambda1=HP_["lambda1"],
                            lambda_ev=HP_["lambda_ev"], scales_dev=sd_dev_,
                            seed=seed)
        _, Ws_ = scin_gpu_([torch.as_tensor(Xe, device=DEVICE, dtype=DTYPE)
                            for Xe in envs_],
                           lambda1=HP_["lambda1"], lambda3=HP_["lambda3"],
                           lambda_ev=HP_["lambda_ev"], scales_dev=sd_dev_,
                           seed=seed, W_init=None)
        st_ = np.stack(Ws_)
        for tau_ in TAU_SWEEP_:
            b_p_ = (np.abs(W_p_) > tau_).astype(float)
            p_, r_, f_ = prf_(W_gt_, b_p_)
            rows2_.append({"m": "NOTEARS-pooled", "tau": tau_,
                           "precision": p_, "recall": r_, "f1": f_,
                           "shd": shd_(W_gt_, b_p_)})
            b_m_ = scin_select_masked_(st_, envs_, tau_, HP_["var_tol"])
            p_, r_, f_ = prf_(W_gt_, b_m_)
            rows2_.append({"m": "SCIN (ours)", "tau": tau_,
                           "precision": p_, "recall": r_, "f1": f_,
                           "shd": shd_(W_gt_, b_m_)})
        print(f"[10-seed] seed {seed} done [{time.time()-t0_:.0f}s]")
    summ2_ = {}
    for m_ in ["NOTEARS-pooled", "SCIN (ours)"]:
        rr_ = [r_ for r_ in rows2_ if r_["m"] == m_]
        best_ = max(rr_, key=lambda r_: r_["f1"])
        f1s_ = [r_["f1"] for r_ in rr_ if r_["tau"] == best_["tau"]]
        precs_ = [r_["precision"] for r_ in rr_ if r_["tau"] == best_["tau"]]
        shds_ = [r_["shd"] for r_ in rr_ if r_["tau"] == best_["tau"]]
        summ2_[m_] = {"best_tau": best_["tau"],
                      "f1": round(float(np.mean(f1s_)), 3),
                      "f1_std": round(float(np.std(f1s_)), 3),
                      "precision": round(float(np.mean(precs_)), 3),
                      "precision_std": round(float(np.std(precs_)), 3),
                      "shd": round(float(np.mean(shds_)), 1)}
    out_["parts"]["part2_chambers_10seed"] = {
        "n_envs": 12, "n_nodes": 18, "n_gt_edges": int(W_gt_.sum()),
        "n_seeds": len(seeds2_), "summary": summ2_,
        "time_s": round(time.time() - t0_, 1)}
except Exception as ex:
    import traceback
    out_["parts"]["part2_chambers_10seed"] = {"error": repr(ex),
                                              "trace": traceback.format_exc()[-1500:]}

# =============================================================================
# PART 3 — linear scale-up extension: d in {300, 400} (synthetic, same
# generator family as the d=25..200 runs: n_edges=1.2d, 2 envs, seed 0)
# =============================================================================
try:
    t0_ = time.time()
    D_LIST_ = [25] if SMOKE else [300, 400]
    hp3_ = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0,
            "tau": 0.2, "var_tol": 0.01}        # gpu_scaleup protocol HPs
    rows3_ = []
    for d_ in D_LIST_:
        n_per_ = 2000
        W_inv_, Xs_np_ = make_env_(d_, int(1.2 * d_), 2, n_per_,
                                   max(3, d_ // 5), seed=0)
        sd_ = np.vstack(Xs_np_).std(0) + 1e-12
        sd_dev_ = torch.as_tensor(sd_, device=DEVICE, dtype=DTYPE)
        Xs_ = [torch.as_tensor(X, device=DEVICE, dtype=DTYPE) for X in Xs_np_]
        Xp_ = torch.vstack(Xs_)
        tw_ = time.time()
        W_pool_ = notears_gpu_(Xp_, lambda1=hp3_["lambda1"],
                               lambda_ev=hp3_["lambda_ev"], scales_dev=sd_dev_,
                               max_iter=100, seed=0)
        tw_ = time.time() - tw_
        ts_ = time.time()
        _, Ws_ = scin_gpu_(Xs_, lambda1=hp3_["lambda1"],
                           lambda3=hp3_["lambda3"], lambda_ev=hp3_["lambda_ev"],
                           scales_dev=sd_dev_, max_iter=150, seed=0,
                           W_init=torch.as_tensor(W_pool_, device=DEVICE,
                                                  dtype=DTYPE))
        ts_ = time.time() - ts_
        st_ = np.stack(Ws_)
        est_ = ((st_.var(0) < hp3_["var_tol"]) &
                (np.abs(st_).mean(0) > hp3_["tau"])).astype(float)
        p_, r_, f_ = prf_(W_inv_, est_)
        rows3_.append({"d": d_, "n_per_env": n_per_,
                       "precision": p_, "recall": r_, "f1": f_,
                       "shd": shd_(W_inv_, est_), "warm_s": round(tw_, 1),
                       "scin_s": round(ts_, 1)})
        print(f"[d-scale] d={d_}: F1={f_:.3f} SHD={rows3_[-1]['shd']} "
              f"warm={tw_:.0f}s scin={ts_:.0f}s [{time.time()-t0_:.0f}s]")
    out_["parts"]["part3_d_300_400"] = {"rows": rows3_,
                                        "time_s": round(time.time()-t0_, 1)}
except Exception as ex:
    import traceback
    out_["parts"]["part3_d_300_400"] = {"error": repr(ex),
                                        "trace": traceback.format_exc()[-1500:]}

out_["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
print("\n==================== COPY THIS JSON BACK ====================")
print(json.dumps(out_, indent=1))
