"""Structure learning models.

- NOTEARS (Zheng et al. 2018): classical differentiable-DAG baseline, linear,
  fitted on pooled data via augmented Lagrangian + L-BFGS.
- SCIN (ours): joint multi-environment fit with per-environment mechanism
  deltas and an invariance penalty on cross-environment coefficient variance.
"""

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- NOTEARS ----

def _h_func(Wm: torch.Tensor) -> torch.Tensor:
    """NOTEARS polynomial acyclicity constraint (numerically stable variant):
    h(W) = tr((I + W o W / d)^d) - d. Same zeros as tr(expm(W o W)) - d but
    cannot overflow, which matters once the augmented-Lagrangian rho grows."""
    d = Wm.shape[0]
    M = torch.eye(d, device=Wm.device) + Wm * Wm / d
    return torch.linalg.matrix_power(M, d).diagonal().sum() - d


def notears_fit(X: np.ndarray, lambda1: float = 0.02, h_tol: float = 1e-8,
                rho_max: float = 1e16, max_outer: int = 100,
                max_inner: int = 200, seed: int = 0,
                lambda_ev: float = 0.0, scales: np.ndarray | None = None) -> np.ndarray:
    """Linear NOTEARS on a single (possibly pooled) dataset. Returns W (d,d).

    lambda_ev > 0 adds an equal-error-variance criterion (Peters & Buhlmann
    2014): a penalty on the cross-node heterogeneity of log residual
    variances, which makes the linear-Gaussian DAG identifiable (default
    NOTEARS on Gaussian data is identifiable only up to the MEC).

    scales: per-column standard deviations used to standardize the data (None
    = ones). The criterion is evaluated in ORIGINAL units, i.e. residual
    variance of column j is var(residual_j) * scales[j]**2; evaluating it in
    standardized units would penalize the true equal-variance DAG.
    """
    d0 = X.shape[1]
    scales = np.ones(d0) if scales is None else np.asarray(scales)
    scales_t = torch.as_tensor(scales ** 2)

    def ev_penalty(Wm, X_t):
        if lambda_ev == 0.0:
            return torch.zeros((), dtype=Wm.dtype)
        r = X_t - X_t @ Wm
        v = r.var(dim=0, unbiased=False) * scales_t
        logv = torch.log(v + 1e-12)
        return lambda_ev * ((logv - logv.mean()) ** 2).sum()
    torch.manual_seed(seed)
    X_t = torch.as_tensor(X)
    n, d = X_t.shape
    eye = torch.eye(d)
    mask = 1.0 - eye

    W = torch.zeros(d, d, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=max_inner, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    state = {"rho": 1.0, "alpha": 0.0, "h": np.inf}

    h_func = _h_func

    for _ in range(max_outer):
        def closure():
            opt.zero_grad()
            Wm = W * mask
            loss = 0.5 / n * torch.sum((X_t - X_t @ Wm) ** 2)
            h = h_func(Wm)
            obj = loss + lambda1 * W.abs().sum() + 0.5 * state["rho"] * h ** 2 \
                + state["alpha"] * h + ev_penalty(Wm, X_t)
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            h = float(h_func(W * mask))
        if h < h_tol or state["rho"] >= rho_max:
            break
        if h > 0.25 * state["h"]:
            state["rho"] *= 10
        else:
            state["alpha"] += state["rho"] * h
        state["h"] = h
    return (W * mask).detach().numpy()


# ------------------------------------------------------------------- SCIN ----

def scin_fit(Xs, lambda1: float = 0.02, lambda3: float = 0.01,
             h_tol: float = 1e-8, rho_max: float = 1e16,
             inner_steps: int = 300, lr: float = 0.05, seed: int = 0,
             W_init: np.ndarray | None = None, lambda_ev: float = 0.0,
             scales: np.ndarray | None = None):
    """Joint multi-environment differentiable DAG learning with invariance.

    Parameters
    - Xs: list of (n_e, d) arrays, one per environment.
    - Shared structural weights W with per-env deltas Delta_e: W_e = W + Delta_e.
    - Objective:
        sum_e 0.5/n_e ||X_e - X_e W_e||_F^2
        + lambda1 ||W||_1
        + lambda3 sum_e ||Delta_e||_F^2        (ridge: ties envs, allows heterogeneity)
        + augmented-Lagrangian DAG terms on the shared W.

    Invariance is used at *selection* time via edges_from_scin: an edge is kept
    only if its per-environment coefficients are consistent (Var_e W_e[i,j]
    below var_tol). lambda3 controls the fit-vs-invariance trade-off in the
    spirit of invariant causal prediction.

    W_init: optional warm start (e.g. pooled NOTEARS solution).

    Returns (W, [W_1..W_E]).
    """
    torch.manual_seed(seed)
    Xs_t = [torch.as_tensor(X) for X in Xs]
    ns = [X.shape[0] for X in Xs]
    d = Xs_t[0].shape[1]
    E = len(Xs_t)
    mask = 1.0 - torch.eye(d)

    W0 = torch.zeros(d, d) if W_init is None else torch.as_tensor(W_init)
    W = W0.clone().requires_grad_(True)
    sc_t = torch.as_tensor(np.ones(d) if scales is None else np.asarray(scales) ** 2)
    Deltas = [torch.zeros(d, d, requires_grad=True) for _ in range(E)]
    params = [W] + Deltas
    opt = torch.optim.LBFGS(params, max_iter=inner_steps,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    state = {"rho": 1.0, "alpha": 0.0, "h": np.inf}

    h_func = _h_func

    for _ in range(100):
        def closure():
            opt.zero_grad()
            Wm = W * mask
            Ws = [(W + D) * mask for D in Deltas]
            recon = sum(
                0.5 / ns[e] * torch.sum((Xs_t[e] - Xs_t[e] @ Ws[e]) ** 2)
                for e in range(E)
            )
            l1 = W.abs().sum()
            ridge = sum((D ** 2).sum() for D in Deltas)
            # equal-variance criterion per environment (original units)
            ev = torch.zeros((), dtype=Wm.dtype)
            if lambda_ev != 0.0:
                for e in range(E):
                    r = Xs_t[e] - Xs_t[e] @ Ws[e]
                    v = r.var(dim=0, unbiased=False) * sc_t
                    logv = torch.log(v + 1e-12)
                    ev = ev + ((logv - logv.mean()) ** 2).sum()
            # acyclicity must hold for EVERY environment's structural matrix
            hs = [h_func(We) for We in Ws]
            h = sum(hs)
            obj = recon + lambda1 * l1 + lambda3 * ridge + lambda_ev * ev \
                + sum(0.5 * state["rho"] * he ** 2 + state["alpha"] * he
                      for he in hs)
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            Ws_ = [(W + D).mul(mask) for D in Deltas]
            h = float(sum(h_func(We) for We in Ws_))
        if h < h_tol or state["rho"] >= rho_max:
            break
        if h > 0.25 * state["h"]:
            state["rho"] *= 10
        else:
            state["alpha"] += state["rho"] * h
        state["h"] = h

    with torch.no_grad():
        Ws_np = [(W + D).mul(mask).numpy() for D in Deltas]
    return W.detach().numpy(), Ws_np


# ------------------------------------------------- graph extraction rules ----

def edges_from_single(W: np.ndarray, tau: float) -> np.ndarray:
    return (np.abs(W) > tau).astype(float)


def edges_from_intersection(Ws: list, tau: float) -> np.ndarray:
    """Edge present iff detected with |w|>tau in every environment."""
    out = None
    for We in Ws:
        b = (np.abs(We) > tau).astype(float)
        out = b if out is None else (out * b)
    return out


def edges_from_scin(Ws: list, tau: float, var_tol: float) -> np.ndarray:
    """Edge present iff magnitudes are consistent across environments
    (Var_e w_e < var_tol) and the mean magnitude exceeds tau."""
    stacked = np.stack(Ws)                       # (E, d, d)
    var = stacked.var(axis=0)
    mag = np.abs(stacked).mean(axis=0)
    return ((var < var_tol) & (mag > tau)).astype(float)
