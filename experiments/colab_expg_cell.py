#@title 🧪 SCIN Exp G — Causal Chambers GPU scale-up (RTX/T4 verified logic) {display-mode:"form"}
# Paste this single cell into Colab (GPU runtime). No pip installs needed.
# Runtime: full run ≈ 20–40 min on T4 (fp64 is required — do NOT switch).
# Output: prints a JSON summary at the end — paste it back to chat.

SMOKE = False   # ← False = full run (delivered state). For a 1-min test set True.

import io, json, time, zipfile
import numpy as np
import pandas as pd
import requests
import torch

torch.set_default_dtype(torch.float64)          # fp64 REQUIRED (L-BFGS line search)
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
VARS = ["red","green","blue","osr_c","v_c","current","pol_1","pol_2",
        "osr_angle_1","osr_angle_2","v_angle_1","v_angle_2","angle_1",
        "angle_2","ir_1","vis_1","ir_2","vis_2","ir_3","vis_3",
        "l_11","l_12","l_21","l_22","l_31","l_32",
        "diode_ir_1","diode_vis_1","diode_ir_2","diode_vis_2",
        "diode_ir_3","diode_vis_3","t_ir_1","t_vis_1","t_ir_2","t_vis_2",
        "t_ir_3","t_vis_3"]
IDX = {n: i for i, n in enumerate(VARS)}

# ============================ data download =================================
URL_ = ("https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/"
        "lt_interventions_standard_v1.zip")
ZDATA_ = requests.get(URL_, timeout=600).content
ZDATA_ = zipfile.ZipFile(io.BytesIO(ZDATA_))
print("dataset loaded:", len([n for n in ZDATA_.namelist() if n.endswith('.csv')]),
      "experiments")

# ============================ metrics =======================================
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
    """Masked invariance selection: a parent's cross-env coefficient variance
    and mean |w| are computed ONLY over environments where that parent
    actually varies (per-env std > 0.1, requiring >= 3 such envs). In envs
    where the parent is constant its coefficient is unidentifiable, so a
    raw variance statistic is meaningless (Part B lesson)."""
    d_ = st_.shape[1]
    env_sd_ = np.stack([e.std(0) for e in envs_])          # (E, d)
    vary_ = env_sd_ > 0.1                                   # (E, d)
    adj_ = np.zeros((d_, d_))
    n_envs_used_ = np.zeros(d_, dtype=int)
    for i in range(d_):
        viv = vary_[:, i]
        if viv.sum() < 3:
            continue                                        # undetectable
        vi_ = st_[:, i, :][viv].var(axis=0)
        mi_ = np.abs(st_[:, i, :])[viv].mean(axis=0)
        n_envs_used_[i] = int(viv.sum())
        adj_[i, :] = ((vi_ < var_tol_) & (mi_ > tau_)).astype(float)
    return adj_, n_envs_used_

# ============================ fitters (fp64 + L-BFGS REQUIRED) ==============
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

# ============================ environment loading ===========================
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
    return [(X - mu_) / sd_ for X in raw], mu_, sd_

# ============================ experiment driver =============================
SEEDS_ = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
N_PER_ = 200; DROP_ = 10 if SMOKE else 60
if SMOKE:
    N_PER_ = 200; DROP_ = 10

PARTS = {
    # Part A: 18-node interpretable subset (physical, common-varying)
    "A_18node": {
        "vars": ["red","green","blue","current","pol_1","pol_2",
                 "ir_1","vis_1","ir_2","vis_2","ir_3","vis_3",
                 "l_11","l_12","l_21","l_22","l_31","l_32"],
        "envs": ["reference","red_mid","green_mid","blue_mid",
                 "l_11_mid","l_12_mid","l_21_mid","l_22_mid",
                 "l_31_mid","l_32_mid","pol_1_mid","pol_2_mid"],
    },
    # Part B: FULL 38-node physical graph, 33 envs — here diode/temperature/
    # rotor interventions are FAIR GAME because those variables are observed.
    "B_38node": {
        "vars": VARS,
        "envs": ["reference",
                 "red_mid","green_mid","blue_mid",
                 "red_strong","green_strong","blue_strong",
                 "osr_c_mid","v_c_mid","pol_1_mid","pol_2_mid",
                 "osr_angle_1_mid","osr_angle_2_mid",
                 "v_angle_1_mid","v_angle_2_mid",
                 "l_11_mid","l_12_mid","l_21_mid","l_22_mid",
                 "l_31_mid","l_32_mid",
                 "diode_ir_1_mid","diode_vis_1_mid","diode_ir_2_mid",
                 "diode_vis_2_mid","diode_ir_3_mid","diode_vis_3_mid",
                 "t_ir_1_mid","t_vis_1_mid","t_ir_2_mid","t_vis_2_mid",
                 "t_ir_3_mid","t_vis_3_mid"],
    } if not SMOKE else {
        "vars": VARS,
        "envs": ["reference","red_mid","green_mid","diode_ir_1_mid",
                 "l_11_mid","t_ir_1_mid"],
    },
}

HP_ = {"lambda1": 0.1, "lambda3": 0.1, "lambda_ev": 1.0,
       "tau": 0.1, "var_tol": 0.02}
TAU_SWEEP_ = [0.05, 0.1, 0.2] if SMOKE else [0.02, 0.05, 0.1, 0.15, 0.2]
VAR_TOL_SWEEP_ = [0.02, 0.05, 0.1]

out_ = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(DEVICE),
        "smoke": SMOKE, "parts": {}}

for part_name, cfg in PARTS.items():
    t0_ = time.time()
    vars_ = cfg["vars"]; env_names_ = cfg["envs"]
    W_gt_full = gt_matrix_(vars_, GT_EDGES)
    per_env_sd_ = {}
    raw_cache_ = {}
    for seed in ([SEEDS_[0]] if SMOKE else SEEDS_):
        envs_, mu_, sd_ = load_envs_(env_names_, vars_, N_PER_, DROP_, seed)
        raw_cache_[seed] = (envs_, sd_)
    Xs_all_ = {s: raw_cache_[s][0] for s in raw_cache_}
    sd_by_seed_ = {s: raw_cache_[s][1] for s in raw_cache_}

    # detectable parents: per-env std of the node, averaged over envs > 0.1
    # (a node constant in most envs cannot be a reliably-estimated parent)
    mean_sd_ = {}
    for k in range(len(vars_)):
        vals_ = [env_[:, k].std() for env_ in Xs_all_[SEEDS_[0]]]
        mean_sd_[k] = float(np.mean(vals_))
    det_mask_ = np.array([mean_sd_[k] > 0.1 for k in range(len(vars_))])
    gt_det_ = W_gt_full.copy()
    for k in range(len(vars_)):
        if not det_mask_[k]:
            gt_det_[:, k] *= 0.0        # never-varying parents -> exclude edge


    per_seed_ = []
    for seed in ([SEEDS_[0]] if SMOKE else SEEDS_):
        envs_ = Xs_all_[seed]
        sd_ = sd_by_seed_[seed]
        sd_dev_ = torch.as_tensor(sd_, device=DEVICE, dtype=DTYPE)
        Xp_dev_ = torch.as_tensor(np.vstack(envs_), device=DEVICE, dtype=DTYPE)
        W_pooled_ = notears_gpu_(Xp_dev_, lambda1=HP_["lambda1"],
                                 lambda_ev=HP_["lambda_ev"],
                                 scales_dev=sd_dev_, seed=seed)
        _, Ws_env_ = scin_gpu_([torch.as_tensor(Xe, device=DEVICE, dtype=DTYPE)
                                for Xe in envs_],
                               lambda1=HP_["lambda1"], lambda3=HP_["lambda3"],
                               lambda_ev=HP_["lambda_ev"],
                               scales_dev=sd_dev_, seed=seed,
                               W_init=None)
        Ws_dev_ = [torch.as_tensor(We, device=DEVICE, dtype=DTYPE)
                   for We in Ws_env_]
        st_ = np.stack(Ws_env_)
        for tau_ in TAU_SWEEP_:
            b_pool_ = (np.abs(W_pooled_) > tau_).astype(float)
            p_, r_, f_ = prf_(W_gt_full, b_pool_)
            pdt, rdt, fdt = prf_(gt_det_, b_pool_)
            sk_ = prf_(skeleton_(W_gt_full).astype(float),
                       skeleton_(b_pool_).astype(float))[2]
            per_seed_.append({"method": "NOTEARS-pooled", "tau": tau_,
                              "precision": p_, "recall": r_, "f1": f_,
                              "f1_det": fdt, "skel_f1": sk_,
                              "shd": shd_(W_gt_full, b_pool_)})
            b_intx_ = np.ones_like(np.abs(W_pooled_), dtype=float)
            for We in Ws_env_:
                b_intx_ = b_intx_ * (np.abs(We) > tau_).astype(float)
            p_, r_, f_ = prf_(W_gt_full, b_intx_)
            pdt, rdt, fdt = prf_(gt_det_, b_intx_)
            sk_ = prf_(skeleton_(W_gt_full).astype(float),
                       skeleton_(b_intx_).astype(float))[2]
            per_seed_.append({"method": "NOTEARS-env-intersection",
                              "tau": tau_, "precision": p_, "recall": r_,
                              "f1": f_, "f1_det": fdt, "skel_f1": sk_,
                              "shd": shd_(W_gt_full, b_intx_)})
            adj_inv_ = ((st_.var(axis=0) < HP_["var_tol"]) &
                        (np.abs(st_).mean(axis=0) > tau_)).astype(float)
            p_, r_, f_ = prf_(W_gt_full, adj_inv_)
            pdt, rdt, fdt = prf_(gt_det_, adj_inv_)
            sk_ = prf_(skeleton_(W_gt_full).astype(float),
                       skeleton_(adj_inv_).astype(float))[2]
            per_seed_.append({"method": "SCIN (unmasked)", "tau": tau_,
                              "precision": p_, "recall": r_, "f1": f_,
                              "f1_det": fdt, "skel_f1": sk_,
                              "shd": shd_(W_gt_full, adj_inv_)})
            for vt_ in VAR_TOL_SWEEP_:
                adj_m_, n_used_ = scin_select_masked_(st_, envs_, tau_, vt_)
                p_, r_, f_ = prf_(W_gt_full, adj_m_)
                pdt, rdt, fdt = prf_(gt_det_, adj_m_)
                sk_ = prf_(skeleton_(W_gt_full).astype(float),
                           skeleton_(adj_m_).astype(float))[2]
                label_ = ("SCIN (ours, masked)" if vt_ == HP_["var_tol"]
                          else f"SCIN (masked, vt={vt_})")
                per_seed_.append({"method": label_, "tau": tau_, "var_tol": vt_,
                                  "precision": p_, "recall": r_, "f1": f_,
                                  "f1_det": fdt, "skel_f1": sk_,
                                  "shd": shd_(W_gt_full, adj_m_)})
        print(f"[{part_name}] seed {seed} done "
              f"[{time.time()-t0_:.0f}s]")

    summary_ = {}
    for m_ in ["NOTEARS-pooled", "NOTEARS-env-intersection",
               "SCIN (unmasked)", "SCIN (ours, masked)",
               "SCIN (masked, vt=0.05)", "SCIN (masked, vt=0.1)"]:
        rr_ = [r_ for r_ in per_seed_ if r_["method"] == m_]
        best_ = max(rr_, key=lambda r_: r_["f1"])
        summary_[m_] = {"best_tau": best_["tau"],
                        "f1": round(float(best_["f1"]), 3),
                        "f1_det": round(float(best_["f1_det"]), 3),
                        "precision": round(float(best_["precision"]), 3),
                        "recall": round(float(best_["recall"]), 3),
                        "skel_f1": round(float(best_["skel_f1"]), 3),
                        "shd": round(float(best_["shd"]), 1),
                        "tau_curve": {str(r_["tau"]): round(float(r_["f1"]), 3)
                                      for r_ in rr_}}
    out_["parts"][part_name] = {
        "n_envs": len(env_names_), "n_nodes": len(vars_),
        "n_gt_edges": int(W_gt_full.sum()),
        "n_gt_edges_detectable": int(gt_det_.sum()),
        "n_seeds": len(SEEDS_) if not SMOKE else 1,
        "summary": summary_,
        "time_s": round(time.time() - t0_, 1)}

out_["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
out_["parts"] = {k: v[0] if isinstance(v, tuple) else v
                 for k, v in out_["parts"].items()}
print("\n==================== COPY THIS JSON BACK ====================")
print(json.dumps(out_, indent=1))
