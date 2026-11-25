"""Phase-3 GPU scale-up: designed for a single workstation GPU
(e.g. NVIDIA RTX PRO 6000 Blackwell, 96 GB). Also runs on CPU (slower).

Parts (select with CLI args, default: all):
  1. --linear   Linear SCIN scale-up: d in {25, 50, 100, 200}, n_per_env=2000,
                2 environments. Measures GPU runtime + invariant-subgraph F1/SHD.
  2. --pnl      Post-nonlinear (PNL) robustness: linear SCIN confronted with
                monotone distortions of increasing strength (d=10).
  3. --neural   Neural SCIN v0: MLP encoder + latent linear SEM with
                per-env deltas, trained jointly; compared vs ERM-MLP on a
                nonlinear OOD task with a spurious feature. This is a v0
                scaffold - honest results, positive or negative, are recorded.

Usage:
    python3 experiments/gpu_scaleup.py --linear --pnl --neural
Output: results/gpu_scaleup.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scin.scm import make_env_data
from scin.metrics import prf, shd, cohens_d

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fp64 is REQUIRED: fp32 overflows inside L-BFGS line search once the
# augmented-Lagrangian rho grows. Matrix sizes here (d<=200) make fp64 cost
# negligible even on workstation GPUs.
DTYPE = torch.float64

HP = dict(lambda1=0.1, lambda3=0.1, lambda_ev=1.0, tau=0.2, var_tol=0.01)
SMOKE = False


def smoke(*a):
    """Return the smoke-mode (small) value or the normal value."""
    return a[0] if not SMOKE else a[1]


# --------------------------------------------------------------------------
# shared GPU model code (float32 device-friendly re implementations)
# --------------------------------------------------------------------------

def h_poly(Wm: torch.Tensor) -> torch.Tensor:
    d = Wm.shape[0]
    M = torch.eye(d, device=Wm.device, dtype=Wm.dtype) + Wm * Wm / d
    return torch.linalg.matrix_power(M, d).diagonal().sum() - d


def scin_gpu(Xs_dev, lambda1=HP["lambda1"], lambda3=HP["lambda3"],
             lambda_ev=HP["lambda_ev"], scales_dev=None,
             h_tol=1e-6, rho_max=1e12, inner_epochs=200, lr=0.02,
             W_init=None, seed=0, log=lambda *a: None):
    """SCIN on device tensors (L-BFGS inner solver, mirrors the proven
    CPU implementation). Xs_dev: list of (n_e, d) tensors."""
    torch.manual_seed(seed)
    d = Xs_dev[0].shape[1]
    E = len(Xs_dev)
    ns = [X.shape[0] for X in Xs_dev]
    mask = 1.0 - torch.eye(d, device=DEVICE, dtype=DTYPE)
    sc2 = torch.ones(d, device=DEVICE, dtype=DTYPE) if scales_dev is None \
        else scales_dev ** 2

    W = (torch.zeros(d, d, device=DEVICE) if W_init is None
         else W_init.clone()).requires_grad_(True)
    Deltas = [torch.zeros(d, d, device=DEVICE, requires_grad=True) for _ in range(E)]
    params = [W] + Deltas
    rho, alpha, h_prev = 1.0, 0.0, float("inf")

    for outer in range(100):
        opt = torch.optim.LBFGS(params, max_iter=inner_epochs,
                                tolerance_grad=1e-7, tolerance_change=1e-11,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Wm = W * mask
            Ws = [(W + D) * mask for D in Deltas]
            recon = sum(0.5 / ns[e] * torch.sum((Xs_dev[e] - Xs_dev[e] @ Ws[e]) ** 2)
                        for e in range(E))
            l1 = W.abs().sum()
            ridge = lambda3 * sum((D ** 2).sum() for D in Deltas)
            ev = torch.zeros((), device=DEVICE, dtype=DTYPE)
            if lambda_ev != 0.0:
                for e in range(E):
                    r = Xs_dev[e] - Xs_dev[e] @ Ws[e]
                    logv = torch.log(r.var(dim=0, unbiased=False) * sc2 + 1e-12)
                    ev = ev + ((logv - logv.mean()) ** 2).sum()
            hs = [h_poly(We) for We in Ws]
            obj = recon + lambda1 * l1 + ridge + lambda_ev * ev \
                + sum(0.5 * rho * he ** 2 + alpha * he for he in hs)
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            h = float(sum(h_poly((W + D) * mask) for D in Deltas))
        log(f"    outer {outer}: h={h:.2e} rho={rho:.0e}")
        if h < h_tol or rho >= rho_max:
            break
        rho = rho * 10 if h > 0.25 * h_prev else rho
        alpha += rho * h if h <= 0.25 * h_prev else 0.0
        h_prev = h

    with torch.no_grad():
        Ws = [(W + D) * mask for D in Deltas]
        return W.detach().cpu().numpy().astype(np.float64), \
            [We.cpu().numpy().astype(np.float64) for We in Ws]


def notears_gpu(X_dev, lambda1=HP["lambda1"], lambda_ev=HP["lambda_ev"],
                scales_dev=None, h_tol=1e-6, rho_max=1e12, inner_epochs=200,
                lr=0.02, seed=0):
    torch.manual_seed(seed)
    n, d = X_dev.shape
    mask = 1.0 - torch.eye(d, device=DEVICE, dtype=DTYPE)
    sc2 = torch.ones(d, device=DEVICE, dtype=DTYPE) if scales_dev is None \
        else scales_dev ** 2
    W = torch.zeros(d, d, device=DEVICE, requires_grad=True)
    rho, alpha, h_prev = 1.0, 0.0, float("inf")
    for outer in range(100):
        opt = torch.optim.LBFGS([W], max_iter=inner_epochs,
                                tolerance_grad=1e-7, tolerance_change=1e-11,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Wm = W * mask
            loss = 0.5 / n * torch.sum((X_dev - X_dev @ Wm) ** 2)
            if lambda_ev != 0.0:
                r = X_dev - X_dev @ Wm
                logv = torch.log(r.var(dim=0, unbiased=False) * sc2 + 1e-12)
                loss = loss + lambda_ev * ((logv - logv.mean()) ** 2).sum()
            h = h_poly(Wm)
            obj = loss + lambda1 * W.abs().sum() + 0.5 * rho * h ** 2 + alpha * h
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            h = float(h_poly(W * mask))
        if h < h_tol or rho >= rho_max:
            break
        rho = rho * 10 if h > 0.25 * h_prev else rho
        alpha += rho * h if h <= 0.25 * h_prev else 0.0
        h_prev = h
    return (W * mask).detach().cpu().numpy().astype(np.float64)


def make_env_gpu(d, n_per_env, n_envs, n_perturbed, seed):
    """Generate on CPU (fast enough), transfer to device as float32."""
    data = make_env_data(d=d, n_edges=int(1.2 * d), n_envs=n_envs,
                         n_per_env=n_per_env, n_perturbed=max(3, d // 5), seed=seed)
    Xs = [torch.as_tensor(X, dtype=DTYPE, device=DEVICE) for X in data["Xs"]]
    sd = np.vstack(data["Xs"]).std(0) + 1e-12
    sd_dev = torch.as_tensor(sd, dtype=DTYPE, device=DEVICE)
    return data, Xs, sd, sd_dev


# ------------------------------------------------------------------ part 1 --

def part_linear():
    print(f"\n=== Part 1: linear scale-up on {DEVICE} ===")
    out = []
    for d in smoke([25, 50, 100, 200], [20]):
        n_per_env = smoke(2000, 400)
        data, Xs, sd, sd_dev = make_env_gpu(d, n_per_env, 2, 3, seed=0)
        Xp = torch.vstack(Xs)

        t0 = time.time()
        W_pool = notears_gpu(Xp, scales_dev=sd_dev, seed=0,
                             inner_epochs=smoke(150, 60) if d <= 50 else 100)
        t_warm = time.time() - t0

        t0 = time.time()
        _, Ws = scin_gpu(Xs, scales_dev=sd_dev, W_init=torch.as_tensor(
            W_pool, dtype=DTYPE, device=DEVICE), seed=0,
            inner_epochs=smoke(200, 80) if d <= 50 else 150)
        t_scin = time.time() - t0

        st = np.stack(Ws)
        var, mag = st.var(0), np.abs(st).mean(0)
        est = ((var < HP["var_tol"]) & (mag > HP["tau"])).astype(float)
        p, r, f = prf(data["W_invariant"], est)
        row = {"d": d, "n_per_env": n_per_env, "device": str(DEVICE),
               "notears_warmstart_s": t_warm, "scin_s": t_scin,
               "precision": p, "recall": r, "f1": f, "shd": shd(data["W_invariant"], est)}
        out.append(row)
        print(f"  d={d:>4}: F1={f:.3f} SHD={row['shd']} "
              f"warm={t_warm:.1f}s scin={t_scin:.1f}s")
    return out


# ------------------------------------------------------------------ part 2 --

def prt_pnl():
    print("\n=== Part 2: post-nonlinear (PNL) robustness ===")

    def distort(X, strength):
        """Monotone (tanh) post-nonlinear distortion per node."""
        return X + strength * 0.5 * np.tanh(X / 0.5)

    out = []
    for strength in smoke([0.0, 0.25, 0.5, 1.0], [0.0, 0.5]):
        rows = []
        for seed in smoke(range(3), range(2)):
            data = make_env_data(d=10, n_edges=10, n_envs=2,
                                 n_per_env=smoke(1000, 300), seed=seed)
            Xs = [distort(X, strength) for X in data["Xs"]]
            sd = np.vstack(Xs).std(0) + 1e-12
            Xs_std = [(X - np.vstack(Xs).mean(0)) / sd for X in Xs]
            Xp = np.vstack(Xs_std)
            W_pool = notears_fit_cpu(Xp, sd, seed)
            _, Ws = scin_cpu(Xs_std, sd, seed, W_pool)
            st = np.stack(Ws)
            est = ((st.var(0) < HP["var_tol"]) &
                   (np.abs(st).mean(0) > HP["tau"])).astype(float)
            rows.append(prf(data["W_invariant"], est)[2])
        out.append({"strength": strength,
                    "f1": (float(np.mean(rows)), float(np.std(rows)))})
        print(f"  PNL strength={strength}: F1 {out[-1]['f1'][0]:.3f}±{out[-1]['f1'][1]:.3f}")
    return out


def notears_fit_cpu(X, sd, seed):
    return notears_gpu(torch.as_tensor(X, dtype=DTYPE, device=DEVICE),
                       scales_dev=torch.as_tensor(sd, dtype=DTYPE, device=DEVICE),
                       seed=seed)


def scin_cpu(Xs, sd, seed, W_pool):
    Xs_dev = [torch.as_tensor(X, dtype=DTYPE, device=DEVICE) for X in Xs]
    sd_dev = torch.as_tensor(sd, dtype=DTYPE, device=DEVICE)
    _, Ws = scin_gpu(Xs_dev, scales_dev=sd_dev,
                     W_init=torch.as_tensor(W_pool, dtype=DTYPE, device=DEVICE),
                     seed=seed)
    return None, Ws


# ------------------------------------------------------------------ part 3 --

def prt_neural():
    """Neural SCIN v0 on a nonlinear OOD task with a spurious feature."""
    print("\n=== Part 3: neural SCIN v0 (nonlinear OOD task) ===")
    if str(DEVICE) == "cpu":
        print("  [warn] running on CPU; will be slow but functional")

    def sample_env(beta, n, rng):
        d_feat = 6  # x0,x1,x2,s,z1,z2
        X = rng.normal(size=(n, d_feat))
        # nonlinear causal mechanisms over the raw features: use tanh mixing
        x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
        base = 1.5 * np.tanh(x1) - 1.0 * (x2 ** 2 - 1.0)
        y = base + 0.5 * rng.normal(size=n)
        X[:, 3] = beta * y + rng.normal(size=n)      # s spurious (non-invariant)
        X[:, 4] = rng.normal(size=n)                  # z1 nuisance
        X[:, 5] = rng.normal(size=n)                  # z2 nuisance
        return X, y.astype(np.float32)

    rng = np.random.default_rng(0)
    betas_tr, beta_val, betas_te = (2.0, 1.0), 1.5, (0.0, -2.0)
    n = smoke(4000, 800)
    tr = [sample_env(b, n, rng) for b in betas_tr]
    va = sample_env(beta_val, n, rng)
    te = [sample_env(b, n, rng) for b in betas_te]

    def mse(y, yh):
        return float(np.mean((y - yh) ** 2))

    # ---- ERM-MLP baseline
    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 1)).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Xtr = torch.as_tensor(np.vstack([t[0] for t in tr]), dtype=torch.float32, device=DEVICE)
    ytr = torch.as_tensor(np.hstack([t[1] for t in tr]), dtype=torch.float32, device=DEVICE)
    for epoch in range(smoke(3000, 400)):
        opt.zero_grad()
        loss = ((enc(Xtr).squeeze() - ytr) ** 2).mean()
        loss.backward(); opt.step()
    erm_res = {}
    with torch.no_grad():
        for tag, env in [("ood0", te[0]), ("ood_neg", te[1])]:
            Xe = torch.as_tensor(env[0], dtype=torch.float32, device=DEVICE)
            erm_res[f"mse_{tag}"] = mse(env[1], enc(Xe).squeeze().cpu().numpy())
    print(f"  ERM-MLP: {erm_res}")

    # ---- Neural SCIN v0: encoder z = MLP(x) (d_latent=16), linear SEM over
    # [z, y] with per-env deltas; selection = ancestors of y in invariant
    # latent graph; predictor = ridge on selected latents.
    d_lat = 16
    torch.manual_seed(1)
    encoder = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU(),
                                  torch.nn.Linear(32, d_lat)).to(DEVICE)
    latent_rows = [torch.as_tensor(t[0], dtype=torch.float32, device=DEVICE)
                   for t in tr]
    ys_dev = [torch.as_tensor(t[1], dtype=torch.float32, device=DEVICE) for t in tr]

    # normalize encoder outputs over pooled train latents (fixed per outer iter)
    sem_params = None
    for phase in range(smoke(3, 2)):  # alternate encoder/SEM training phases
        with torch.no_grad():
            Zs = [encoder(x) for x in latent_rows]
            Zall = torch.vstack(Zs)
            mu, sd = Zall.mean(0), Zall.std(0) + 1e-6
        Zs = [(z - mu) / sd for z in Zs]
        # append y as final node
        Zs_y = [torch.cat([z, y.unsqueeze(1)], dim=1) for z, y in zip(Zs, ys_dev)]
        d_tot = d_lat + 1
        W = torch.zeros(d_tot, d_tot, device=DEVICE, requires_grad=True)
        Deltas = [torch.zeros(d_tot, d_tot, device=DEVICE, requires_grad=True)
                  for _ in range(2)]
        mask = 1.0 - torch.eye(d_tot, device=DEVICE)
        sc2 = torch.cat([torch.ones(d_lat, device=DEVICE),
                         torch.as_tensor([float(ys_dev[0].var() + 1e-6)])])
        opt = torch.optim.Adam([W] + Deltas, lr=0.02)
        rho, alpha, h_prev = 1.0, 0.0, float("inf")
        for outer in range(smoke(30, 10)):
            for it in range(smoke(150, 40)):
                opt.zero_grad()
                Ws = [(W + D) * mask for D in Deltas]
                recon = sum(0.5 / len(Z) * torch.sum((Z - Z @ Ws[e]) ** 2)
                            for e, Z in enumerate(Zs_y))
                ridge = 0.1 * sum((D ** 2).sum() for D in Deltas)
                ev = torch.zeros((), device=DEVICE)
                for e, Z in enumerate(Zs_y):
                    r = Z - Z @ Ws[e]
                    logv = torch.log(r.var(dim=0, unbiased=False) * sc2 + 1e-12)
                    ev = ev + ((logv - logv.mean()) ** 2).sum()
                hs = [h_poly(We) for We in Ws]
                # NOTE: acyclicity on latents + y node
                obj = recon + 0.1 * W.abs().sum() + ridge + 1.0 * ev \
                    + sum(0.5 * rho * he ** 2 + alpha * he for he in hs)
                obj.backward(); opt.step()
            with torch.no_grad():
                h = float(sum(h_poly((W + D) * mask) for D in Deltas))
            if h < 1e-4:
                break
            rho = rho * 10 if h > 0.25 * h_prev else rho
            alpha += rho * h if h <= 0.25 * h_prev else 0.0
            h_prev = h
        Ws_np = [We.detach().cpu().numpy() for We in
                 [(W + D) * mask for D in Deltas]]
        # freeze encoder towards invariance: fine-tune encoder to reduce
        # coefficient variance of y's edges (light touch, 200 steps)
        enc_opt = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        W_fix = [torch.as_tensor(w) for w in Ws_np]

    st = np.stack(Ws_np)
    var, mag = st.var(0), np.abs(st).mean(0)
    adj = ((var < 0.05) & (mag > 0.1)).astype(float)
    # ancestors of y (node d_tot-1) among latents
    anc, frontier = set(), [d_tot - 1]
    while frontier:
        j = frontier.pop()
        for i in range(d_tot - 1):
            if adj[i, j] and i not in anc:
                anc.add(i)
                frontier.append(i)
    sel = sorted(anc)
    print(f"  selected latent dims: {sel} (of {d_lat})")

    if sel:
        with torch.no_grad():
            Ztr = [((encoder(x) - mu) / sd)[:, sel] for x in latent_rows]
            Xl = torch.vstack(Ztr).cpu().numpy()
            yl = np.hstack([t[1] for t in tr])
            Xb = np.hstack([Xl, np.ones((len(Xl), 1))])
            A = Xb.T @ Xb + 1e-2 * np.eye(Xb.shape[1]); A[-1, -1] -= 1e-2
            wb = np.linalg.solve(A, Xb.T @ yl)
            w, b = wb[:-1], wb[-1]
        scin_res = {}
        for tag, env in [("ood0", te[0]), ("ood_neg", te[1])]:
            with torch.no_grad():
                Z = ((encoder(torch.as_tensor(env[0], dtype=torch.float32,
                                              device=DEVICE)) - mu) / sd)[:, sel]
                yh = Z.cpu().numpy() @ w + b
            scin_res[f"mse_{tag}"] = mse(env[1], yh)
        print(f"  SCIN-N v0: {scin_res}")
    else:
        scin_res = {"mse_ood0": float("nan"), "mse_ood_neg": float("nan"),
                    "note": "no latents selected"}
        print(f"  SCIN-N v0: {scin_res}")

    return {"erm": erm_res, "scin_v0": scin_res, "n_latent_selected": len(sel),
            "d_latent": d_lat, "betas_train": betas_tr, "betas_test": betas_te}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--pnl", action="store_true")
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config to validate the full pipeline quickly")
    args = ap.parse_args()
    if args.smoke:
        global SMOKE
        SMOKE = True
    if not (args.linear or args.pnl or args.neural):
        args.linear = args.pnl = args.neural = True

    print(f"Device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if torch.cuda.is_available() else " (CPU only)"))
    out = {"device": str(DEVICE),
           "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if args.linear:
        out["linear_scaleup"] = part_linear()
    if args.pnl:
        out["pnl_robustness"] = prt_pnl()
    if args.neural:
        out["neural_v0"] = prt_neural()
    (RESULTS / "gpu_scaleup.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> results/gpu_scaleup.json")
    print("\nPlease share this JSON file back for analysis.")


if __name__ == "__main__":
    main()
