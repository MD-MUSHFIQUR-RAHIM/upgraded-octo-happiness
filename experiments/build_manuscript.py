"""Build paper/main.tex from PAPER.md via pandoc.

Extracts the Abstract and Main text (Introduction .. Methods, including
Results) sections, converts each to LaTeX (pandoc markdown->latex), and
assembles a compilable single-column article with figure placeholders
(figure files referenced from results/figures/; adjust in Overleaf).
Also converts the Supplementary / figure plan as comments for reference.

Run:  python3 experiments/build_manuscript.py
Writes paper/main.tex. Wiki tables/inline math ($...$, $\\lambda$) are
supported by pandoc. Compilation requires a LaTeX toolchain (Overleaf).
"""
import os, re, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PAPER_MD = os.path.join(ROOT, "PAPER.md")
OUT_DIR = os.path.join(ROOT, "paper")

TITLE = ("Sparse Causal Intervention Networks: Learning Invariant "
         "Mechanisms from Multiple Environments for Distribution-Shift "
         "Generalization")


def md_to_latex(md: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(md)
        src = f.name
    tex = subprocess.run(
        ["python3", "-c",
         "import pypandoc,sys; print(pypandoc.convert_file(sys.argv[1],"
         "'latex', format='markdown', extra_args=['--wrap=preserve']))",
         src],
        capture_output=True, text=True, check=True)
    os.unlink(src)
    return tex.stdout


def section(md: str, header: str, next_headers: list) -> str:
    """Extract text between `## header` and the next header."""
    pat = re.compile(r"^## " + re.escape(header) + r".*$", re.M)
    m = pat.search(md)
    if m is None:
        raise SystemExit(f"header not found: {header}")
    start = m.end()
    ends = [md.find("\n## ", start)]
    for nh in next_headers:
        m2 = re.search(r"^## " + re.escape(nh) + r".*$", md[start:], re.M)
        if m2:
            ends.append(start + m2.start())
    end = min(e for e in ends if e != -1)
    return md[start:end]


def polish(tex: str) -> str:
    """Promote pandoc's heading levels to a proper article structure."""
    import re
    # top-level sections
    tex = re.sub(
        r"\\subsubsection\{\d+\.\s*([A-Za-z ]+)[^}]*}\\label\{[^}]*}",
        r"\\section{\1}", tex)
    # results subsections: "Results 2.x --- Title (...)" -> \subsection{2.x Title}
    def repl(m):
        rest = re.sub("[ ]*(\\([^)]*\\))?[ ]*$", "", m.group(2))
        return "\\subsection{" + m.group(1) + " " + rest + "}" \
               + m.group(3)
    tex = re.sub(
        r"\\paragraph\{Results (2\.\d) --- ([^}]*)}(\\label\{[^}]*})",
        repl, tex)
    tex = tex.replace("\\subsubsection{Supplementary}\\label{supplementary}",
                      "\\section*{Supplementary overview}")
    return tex


def main():
    md = open(PAPER_MD).read()
    abstract = section(md, "Abstract", ["Main text"])
    main_text = section(md, "Main text", ["Supplementary", "Figure plan"])

    def transliterate(t: str) -> str:
        """pdfLaTeX-safe substitutions for combining-diacritic symbols."""
        repl = {
            "β̄": r"$\bar{\beta}$",
            "σ̂": r"$\hat{\sigma}$",
            "Σ̂": r"$\hat{\Sigma}$",
            "Å": r"\AA{}",
            "Erdős–Rényi": "Erdos-Renyi",
            "Bühlmann": "Buhlmann",
            "Sch\u00f6lkopf": "Schoelkopf",
            "Pe'er": "Pe'er",
            "≈": r"$\approx$",
            "×": r"$\times$",
            "±": r"$\pm$",
            "≤": r"$\leq$",
            "≥": r"$\geq$",
            "→": r"$\rightarrow$",
            "←": r"$\leftarrow$",
            "λ": r"$\lambda$",
            "τ": r"$\tau$",
            "γ": r"$\gamma$",
            "σ": r"$\sigma$",
            "β": r"$\beta$",
            "ρ": r"$\rho$",
            "Δ": r"$\Delta$",
            "Σ": r"$\Sigma$",
            "⁻¹": r"$^{-1}$",
            "²": r"$^2$",
            "Ŵ": r"$\hat{W}$",
            "η": r"$\eta$",
            "‖": r"$\Vert$",
            "†": r"$\dagger$",
            "′": r"$'$",
            "₁": "_1",
            "₃": "_3",
            "⇒": r"$\Rightarrow$",
            "∈": r"$\in$",
            "−": "-",
            "∗": r"$\ast$",
            "∘": r"$\circ$",
            "√": r"$\sqrt{x}$",
            "≠": r"$\neq$",
        }
        for k, v in repl.items():
            t = t.replace(k, v)
        return t

    abstract_tex = polish(md_to_latex(transliterate(abstract)))
    body_tex = polish(md_to_latex(transliterate(main_text)))

    figures = [
        ("fig8_concept", "Concept schematic: environment-dependent mechanism "
         "moves around an invariant core; SCIN estimates the invariant set."),
        ("fig1_scaling", "Dimension scaling on GPU: directed F1 and precision "
         "at $d=25..400$ (RTX PRO 6000 Blackwell, fp64)."),
        ("fig3_ood_bars", "OOD error under spurious shift: SCIN matches the "
         "oracle; ERM and V-REx pay the spurious term (Exp B, Exp E)."),
        ("fig4_ablation_violation", "Ablation hierarchy (a) and A1-violation "
         "robustness (b)."),
        ("fig5_icp", "Node-wise ICP power collapse vs SCIN (Exp F)."),
        ("fig6_e_sens", "Selection-sensitivity plateau, var_tol x tau (Exp E)."),
        ("fig7_chambers_adjacency", "Causal Chambers: ground-truth vs SCIN vs "
         "pooled adjacency (18 nodes, seed 0)."),
        ("fig2_pnl", "Post-nonlinear robustness: F1 vs distortion strength."),
        ("fig9_e_scaling", "Environment-count ladder on chambers ($E=4..17$)."),
        ("fig10_exp_h", "Experiment H: controlled env-ladder regimes."),
        ("fig11_sachs", "Sachs domain pilot: sensitivity grid and label-free "
         "stability rule."),
    ]
    fig_block = "".join(
        "\\begin{figure}[htbp]\\centering\n"
        f"\\includegraphics[width=0.9\\textwidth]{{results/figures/{fn}.png}}\n"
        "\\caption{"
        f"{cap} (paper Fig. {i + 1}; file {fn}.png)}}\n"
        "\\label{fig:" + fn + "}\n\\end{figure}\n\n"
        for i, (fn, cap) in enumerate(figures))

    doc = rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{natbib}}
% unicode tolerance for pdfLaTeX (markdown source uses typed symbols)
\DeclareUnicodeCharacter{{00D7}}{{\ensuremath{{\times}}}}
\DeclareUnicodeCharacter{{00B1}}{{\ensuremath{{\pm}}}}
\DeclareUnicodeCharacter{{2264}}{{\ensuremath{{\leq}}}}
\DeclareUnicodeCharacter{{2265}}{{\ensuremath{{\geq}}}}
\DeclareUnicodeCharacter{{2248}}{{\ensuremath{{\approx}}}}
\DeclareUnicodeCharacter{{2212}}{{\ensuremath{{-}}}}
\DeclareUnicodeCharacter{{2192}}{{\ensuremath{{\rightarrow}}}}
\DeclareUnicodeCharacter{{2190}}{{\ensuremath{{\leftarrow}}}}
\DeclareUnicodeCharacter{{2032}}{{\ensuremath{{'}}}}
\DeclareUnicodeCharacter{{03BB}}{{\ensuremath{{\lambda}}}}
\DeclareUnicodeCharacter{{03C4}}{{\ensuremath{{\tau}}}}
\DeclareUnicodeCharacter{{03B3}}{{\ensuremath{{\gamma}}}}
\DeclareUnicodeCharacter{{03C3}}{{\ensuremath{{\sigma}}}}
\DeclareUnicodeCharacter{{03B2}}{{\ensuremath{{\beta}}}}
\DeclareUnicodeCharacter{{03C1}}{{\ensuremath{{\rho}}}}
\DeclareUnicodeCharacter{{0394}}{{\ensuremath{{\Delta}}}}
\DeclareUnicodeCharacter{{03A3}}{{\ensuremath{{\Sigma}}}}
\DeclareUnicodeCharacter{{00B9}}{{\ensuremath{{^1}}}}
\DeclareUnicodeCharacter{{00B2}}{{\ensuremath{{^2}}}}
\newtheorem{{theorem}}{{Theorem}}
\newtheorem{{lemma}}{{Lemma}}
\newtheorem{{corollary}}{{Corollary}}
\newtheorem{{assumption}}{{Assumption}}

\title{{{TITLE}}}
\author{{Mushfiqur Rahim\thanks{{
All code, raw results and this manuscript source are archived in the
project repository; figures regenerate from committed scripts.}}}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{abstract_tex}
\end{{abstract}}

\section*{{Main text}}
{body_tex}

\clearpage
{fig_block}
\end{{document}}
"""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "main.tex")
    open(out, "w").write(doc)
    print("wrote", out, len(doc), "bytes;",
          body_tex.count("\\section"), "sections;",
          "abstract", len(abstract_tex), "chars")


if __name__ == "__main__":
    main()
