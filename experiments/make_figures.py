"""Paper figures for the SCIN project.

Generates publication-style figures from the result JSONs into
results/figures/. Run from the repo root:  python experiments/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})

C_SCIN = "#1f77b4"
C_ERM = "#d62728"
C_ORA = "#2ca02c"
C_REX = "#9467bd"


def jload(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def gpu_data():
    """The real Blackwell GPU run (user-supplied), with CPU fallback."""
    try:
        return jload("gpu_scaleup_gpu.json")
    except FileNotFoundError:
        return jload("gpu_scaleup.json")


def fig1_scaling():
    """F1 vs dimension (GPU run) + F1 vs n (phase-2)."""
    gpu = gpu_data()["linear_scaleup"]
    p2 = jload("phase2_ablations.json")["scaling_n"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    ds = [r["d"] for r in gpu]
    f1s = [r["f1"] for r in gpu]
    axes[0].plot(ds, f1s, "o-", color=C_SCIN, lw=2)
    for d_, r in zip(ds, gpu):
        axes[0].annotate(f"{r['f1']:.2f}", (d_, r["f1"]),
                         textcoords="offset points", xytext=(0, 7), ha="center")
    axes[0].set_xlabel("dimension $d$")
    axes[0].set_ylabel("invariant-subgraph F1")
    axes[0].set_title("(a) scale-up (GPU, $n$=2000/env)")
    axes[0].set_ylim(0.8, 1.02)

    ns = sorted(int(k) for k in p2)
    scin = [p2[str(n)]["scin_f1"][0] for n in ns]
    notears = [p2[str(n)]["notears_f1"][0] for n in ns]
    axes[1].plot(ns, scin, "o-", color=C_SCIN, lw=2, label="SCIN")
    axes[1].plot(ns, notears, "s--", color=C_ERM, lw=2, label="NOTEARS pooled")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("samples per environment $n$")
    axes[1].set_ylabel("F1")
    axes[1].set_title("(b) sample-size scaling")
    axes[1].legend(frameon=False)
    fig.savefig(os.path.join(FIGS, "fig1_scaling.png"))
    plt.close(fig)


def fig2_pnl():
    pnl = gpu_data()["pnl"]
    xs = [r["strength"] for r in pnl]
    ys = [r["f1_mean"] for r in pnl]
    es = [r["f1_std"] for r in pnl]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=C_SCIN, lw=2, capsize=4)
    ax.set_xlabel("post-nonlinear distortion strength")
    ax.set_ylabel("F1")
    ax.set_title("PNL robustness (3 seeds)")
    ax.set_ylim(0.8, 1.0)
    fig.savefig(os.path.join(FIGS, "fig2_pnl.png"))
    plt.close(fig)


def fig3_ood_bars():
    """Exp B (child spurious) + Exp E (confounded) OOD comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    b = jload("exp_b_ood.json")["summary"]
    models = ["ERM", "V-REx", "SCIN (ours)"]
    vals = [b[m]["mse_ood_neg"][0] for m in models]
    errs = [b[m]["mse_ood_neg"][1] for m in models]
    cols = [C_ERM, C_REX, C_SCIN]
    bars = axes[0].bar(["ERM", "V-REx", "SCIN"], vals, yerr=errs,
                       capsize=4, color=cols)
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].text(2.4, 1.0, "oracle = noise\nfloor (1.0)", fontsize=9,
                 va="bottom", ha="right")
    axes[0].set_ylabel("OOD MSE (neg. shift)")
    axes[0].set_title("(a) Exp B: spurious child ($s \\leftarrow y$)")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v:.1f}", ha="center", fontsize=10)

    e = jload("exp_e_confounded.json")
    models = ["ERM", "SCIN", "Oracle"]
    means = [e["mean"]["erm_ood"][1], e["mean"]["scin_ood"][1],
             e["mean"]["oracle_ood"][1]]
    stds = [e["std"]["erm_ood"][1], e["std"]["scin_ood"][1],
            e["std"]["oracle_ood"][1]]
    bars = axes[1].bar(models, means, yerr=stds, capsize=4,
                       color=[C_ERM, C_SCIN, C_ORA])
    axes[1].set_ylabel("OOD MSE ($\\gamma$ sign-flip)")
    axes[1].set_title("(b) Exp E: spurious INSIDE mechanism ($s \\to y$)")
    for bar, v in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                     f"{v:.2f}", ha="center", fontsize=10)
    fig.savefig(os.path.join(FIGS, "fig3_ood_bars.png"))
    plt.close(fig)


def fig4_ablation_violation():
    """(a) component ablations; (b) A1-violation robustness."""
    p2 = jload("phase2_ablations.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))

    ab = p2["ablation_a"]
    names = list(ab)
    f1s = [ab[k]["f1"][0] for k in names]
    errs = [ab[k]["f1"][1] for k in names]
    short = {"SCIN (full)": "SCIN\n(full)",
             "SCIN w/o invariance stat": "w/o inv.\nstat",
             "SCIN w/o per-env deltas": "w/o per-env\ndeltas",
             "SCIN w/o equal-variance crit.": "w/o ev\ncriterion",
             "NOTEARS-pooled": "NOTEARS\npooled"}
    labels = [short.get(k, k) for k in names]
    cols = [C_SCIN] + [C_ERM] * (len(names) - 1)
    axes[0].bar(labels, f1s, yerr=errs, capsize=4, color=cols)
    axes[0].set_ylabel("F1")
    axes[0].set_title("(a) component ablations (5 seeds)")
    axes[0].set_ylim(0, 1.05)

    a1 = p2["a1_violation"]
    jitters = sorted(a1, key=float)
    scin_y = [a1[j]["scin_f1"][0] for j in jitters]
    notears_y = [a1[j]["notears_f1"][0] for j in jitters]
    axes[1].plot([float(j) for j in jitters], scin_y, "o-", color=C_SCIN,
                 lw=2, label="SCIN")
    axes[1].plot([float(j) for j in jitters], notears_y, "s--", color=C_ERM,
                 lw=2, label="NOTEARS pooled")
    axes[1].set_xlabel("noise-variance jitter $j$ (A1 violation)")
    axes[1].set_ylabel("F1")
    axes[1].set_title("(b) robustness to A1 violation")
    axes[1].legend(frameon=False)
    fig.savefig(os.path.join(FIGS, "fig4_ablation_violation.png"))
    plt.close(fig)


def fig6_e_sens():
    """Exp E sensitivity heatmap: OOD_neg over (var_tol, tau); 'S!' marks
    cells where the spurious s gets selected."""
    import numpy as np
    import matplotlib.colors as mcolors
    d = jload("exp_e_sensitivity.json")
    grid = d["grid_mean"]
    vts = d["config"]["var_tols"]
    taus = d["config"]["taus"]
    M = np.zeros((len(vts), len(taus)))
    S = np.zeros_like(M)
    for a, vt in enumerate(vts):
        for b, t in enumerate(taus):
            c = grid[f"{vt}|{t}"]
            M[a, b] = c["ood_neg_mean"]
            S[a, b] = 1.0 if c["s_included"] else 0.0
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(taus)), [str(t) for t in taus])
    ax.set_yticks(range(len(vts)), [str(v) for v in vts])
    ax.set_xlabel("invariance threshold tau")
    ax.set_ylabel("variance tolerance var_tol")
    for a in range(len(vts)):
        for b in range(len(taus)):
            ax.text(b, a, f"{M[a, b]:.2f}" + (" S!" if S[a, b] else ""),
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="OOD_neg MSE")
    ax.set_title("Exp E selection-sensitivity (5 seeds)")
    fig.savefig(os.path.join(FIGS, "fig6_e_sens.png"))
    plt.close(fig)


def fig5_icp():
    """ICP (node-wise subset search) vs SCIN at 2 and 4 environments."""
    data = {}
    for nv in (2, 4):
        s = jload(f"exp_f_icp_baseline_nenvs{nv}.json")["summary"]
        data[nv] = {
            "icp": [s["ICP (node-wise, F-test)"]["f1"],
                    s["ICP (node-wise, F-test)"]["shd"]],
            "scin": [s["SCIN (ours)"]["f1"], s["SCIN (ours)"]["shd"]]}
    envs = sorted(data)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    w = 0.35
    for off, key, lab, col in ((-w / 2, "icp", "ICP (node-wise)", C_ERM),
                               (w / 2, "scin", "SCIN (ours)", C_SCIN)):
        means = [data[nv][key][0][0] for nv in envs]
        stds = [data[nv][key][0][1] for nv in envs]
        bars = ax.bar([x + off for x in range(len(envs))], means,
                      yerr=stds, width=w, color=col, label=lab, capsize=4)
        for b, m, s_ in zip(bars, means, stds):
            ax.annotate(f"{m:.2f}", (b.get_x() + w / 2, m + s_ + 0.02),
                        ha="center", fontsize=9)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels([f"{e} envs" for e in envs])
    ax.set_ylabel("invariant-subgraph F1")
    ax.set_ylim(0, 1.12)
    ax.set_title("ICP subset search vs SCIN (Exp F)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.savefig(os.path.join(FIGS, "fig5_icp.png"))
    plt.close(fig)


def fig7_chambers_adjacency():
    """Exp G: adjacency heatmaps — GT vs SCIN vs NOTEARS-pooled
    (one representative seed-0 window at tau=0.1, var_tol=0.02)."""
    import numpy as np
    from matplotlib.colors import ListedColormap
    d = jload("exp_g_matrices.json")
    vars_ = d["config"]["vars"]
    n = len(vars_)

    def adj(elist):
        A = np.zeros((n, n))
        for it in elist:
            A[it["i"], it["j"]] = 1.0
        return A

    A_gt = adj(d["adj_gt"])
    A_sc = adj(d["adj_scin"])
    A_pl = adj(d["adj_pool"])

    # encoding: 0 nothing; 1 true positive (in GT and selected);
    # 2 false addition (selected, not in GT); 3 miss (in GT, not selected)
    def encode(A):
        E = np.zeros((n, n))
        E[(A_gt > 0) & (A > 0)] = 1.0
        E[(A_gt == 0) & (A > 0)] = 2.0
        E[(A_gt > 0) & (A == 0)] = 3.0
        return E

    encs = [("Ground truth", A_gt * 1.0), ("SCIN (ours)", encode(A_sc)),
            ("NOTEARS-pooled", encode(A_pl))]
    cmap = ListedColormap(["#f2f2f2", C_SCIN, C_ERM, "#e0a800"])
    short = [v.replace("ir_", "ir").replace("vis_", "vs")
             .replace("pol_", "po").replace("l_", "l") for v in vars_]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (title, E) in zip(axes, encs):
        ax.imshow(E, cmap=cmap, vmin=0, vmax=3, interpolation="none")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(title, fontsize=11)
        tp = int((E == 1).sum()); fa = int((E == 2).sum())
        ms = int((E == 3).sum())
        if title != "Ground truth":
            ax.set_xlabel(f"TP {tp} · false+ {fa} · missed {ms}", fontsize=9)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=C_SCIN, label="true positive"),
               mpatches.Patch(color=C_ERM, label="false addition"),
               mpatches.Patch(color="#e0a800", label="missed (in GT)")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8,
                    loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Causal Chambers light tunnel: selected adjacency at "
                 "tau=0.1 (GT 37 edges)", y=1.04)
    fig.savefig(os.path.join(FIGS, "fig7_chambers_adjacency.png"))
    plt.close(fig)


def fig8_concept():
    """Fig 1 concept schematic: environments -> shared+deltas ->
    invariance selection -> invariant DAG -> OOD generalization."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    ax.axis("off"); ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.6)

    def box(x, y, w, h, text, fc="#eef3fb", ec=C_SCIN, fs=9.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#444444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14, color=color,
                     lw=1.3))

    box(0.15, 2.6, 2.2, 1.5,
        "Multiple\nenvironments\n$X^1,\\dots,X^E$", fs=11)
    box(3.0, 2.6, 2.9, 1.5,
        "Differentiable fit\n$W_e = W + \\Delta_e$ per env\n+ per-env acyclicity", fs=8.5)
    box(3.0, 0.4, 2.9, 1.4,
        "Equal-variance\ncriterion (orientation)", fc="#fdf3ef",
        ec=C_ERM, fs=8.5)
    box(6.6, 2.6, 2.7, 1.5,
        "Invariance selection\n$\\mathrm{Var}_e(\\hat W_{e,ij}) < v$\n& mean $|\\hat W| > \\tau$", fs=8.5)
    box(9.9, 2.6, 2.2, 1.5,
        "Invariant DAG\n$\\to$ OOD-stable\nprediction", fc="#eef7ee",
        ec=C_ORA, fs=9.5)
    arrow(2.35, 3.35, 2.95, 3.35)
    arrow(4.45, 2.55, 4.45, 1.85)
    arrow(5.95, 3.35, 6.55, 3.35)
    arrow(7.95, 2.55, 7.95, 1.85)
    arrow(4.45, 1.1, 7.5, 2.55)
    arrow(9.35, 3.35, 9.85, 3.35)

    # moved-mechanism inset (bottom right)
    ax.text(10.2, 1.7, "moved mechanism:", fontsize=8.5, style="italic")
    for yy, lab, col in [(1.15, "env 1:  $x_s \\to y$,  γ=0.8",
                          C_ERM),
                         (0.75, "env 2:  $x_s \\to y$,  γ=0.0",
                          C_ERM),
                         (0.35, "$x_1 \\to y$,  γ=1.5 (all envs)",
                          C_SCIN)]:
        ax.text(10.2, yy, lab, fontsize=9, color=col)
    ax.text(10.2, 0.02, "kept (invariant)", fontsize=8, color=C_SCIN)
    fig.suptitle("SCIN: joint estimation separates invariant mechanisms "
                 "from environment-moved ones", y=0.02, fontsize=11)
    fig.savefig(os.path.join(FIGS, "fig8_concept.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_scaling()
    fig2_pnl()
    fig3_ood_bars()
    fig4_ablation_violation()
    try:
        fig5_icp()
    except FileNotFoundError as e:
        print("fig5 skipped:", e)
    try:
        fig6_e_sens()
    except FileNotFoundError as e:
        print("fig6 skipped:", e)
    try:
        fig7_chambers_adjacency()
    except FileNotFoundError as e:
        print("fig7 skipped:", e)
    try:
        fig8_concept()
    except FileNotFoundError as e:
        print("fig8 skipped:", e)
    print("figures written to", FIGS)
    for f in sorted(os.listdir(FIGS)):
        print("  ", f)
