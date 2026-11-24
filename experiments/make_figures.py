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


if __name__ == "__main__":
    fig1_scaling()
    fig2_pnl()
    fig3_ood_bars()
    fig4_ablation_violation()
    print("figures written to", FIGS)
    for f in sorted(os.listdir(FIGS)):
        print("  ", f)
