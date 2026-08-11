"""Post-process pandoc output: heading levels, labels, and figure insertion.

Pandoc maps '##' to \\subsection because '#' was consumed as the title, so the
hierarchy is lifted by one level; numbered headings like '4.1' are then demoted
back to subsections and their manual numbers stripped so LaTeX numbers them.
"""

import re
import sys

ZH = "--zh" in sys.argv
OUT = "body_zh.tex" if ZH else "body.tex"

FIGS = {
    "FIG1": ("fig1_elastic_fields.pdf", 1.0,
             r"Solute--dislocation interaction from the elastic dipole tensor. "
             r"(a) angular dependence at $r=10$\,\AA; the isotropic Stroh "
             r"solution reproduces the closed form $A\sin\theta/r$ to "
             r"$2.6\times10^{-8}$. (b) anisotropic field of the edge "
             r"dislocation. (c) ratio of anisotropic to isotropic: the "
             r"amplitude changes by only $-1.8\%$ but the angular shape "
             r"deviates by up to $\pm30\%$ near the glide plane, which is "
             r"where $\mathrm{d}U/\mathrm{d}x$ controls pinning."),
    "FIG7": ("fig7_atmosphere_2d.pdf", 1.0,
             r"Two-dimensional Cottrell atmosphere from conserved Kawasaki "
             r"Monte Carlo at 600\,K, $c_0=2$\,at.\%. (a) elastic field with "
             r"the mean-field $c=1/2$ isoline, a circle of diameter $R_c$ "
             r"tangent to the glide plane. (b) equilibrium occupancy. "
             r"(c) radial profile: Monte Carlo exceeds the mean-field "
             r"Fermi--Dirac profile in the transition shell because of "
             r"solute--solute attraction. (d) the Warren--Cowley parameter "
             r"forms an \emph{annulus}, vanishing in the saturated core."),
    "FIG8": ("fig8_warren_cowley.pdf", 0.60,
             r"Warren--Cowley $\alpha_1$ against local composition. The order "
             r"parameter vanishes at both $c\to0$ and $c\to1$: a saturated "
             r"atmosphere carries no chemical order because it contains no "
             r"chemistry. Monte Carlo points lie above the homogeneous "
             r"quasi-chemical curve because a sampling window straddling the "
             r"concentration gradient registers spurious order."),
    "FIG2": ("fig2_edge_depinning.pdf", 1.0,
             r"Edge-dislocation depinning. (a) critical stress against ageing "
             r"fraction at $N_z=128$; the grey band above $\mu/30$ is "
             r"mechanically unattainable, so a fully aged atmosphere admits no "
             r"breakaway. (b) solute--solute interaction raises $\tau_c$ by up "
             r"to $1.8\times$ over the Bernoulli approximation."),
    "FIG5": ("fig5_screw_kmc.pdf", 1.0,
             r"(a) the atmosphere does not pin the rate-controlling screw: "
             r"$\tau_c^{\rm screw}/\tau_c^{\rm edge}<10^{-3}$ at every ageing "
             r"fraction. (b) kink-pair kinetic Monte Carlo: a random Lu field "
             r"raises the flow stress at every temperature from 100 to 450\,K."),
    "FIG6": ("fig6_kappa_threshold.pdf", 1.0,
             r"(a) a phenomenological chemical softening $\kappa$ reverses the "
             r"sign only above $\kappa^*=0.17$\,eV at 150\,K; at 250\,K no "
             r"value suffices because pure Ta is already athermal. (b) the "
             r"annealed fluctuation term $\sigma^2/2k_BT$ against the quenched "
             r"extreme-value limit and against $\kappa^*$."),
    "FIG3": ("fig3_convergence.pdf", 1.0,
             r"Convergence. (a) grid, stress-ramp and cutoff parameters "
             r"converge to better than 1\%. (b) line length does not and "
             r"should not: weakest-link statistics give a logarithmic decay "
             r"with no plateau to $N_z=2048$. (c) realisation scatter of 9.6\% "
             r"sets two significant figures as the reportable precision."),
    "FIG4": ("fig4_sensitivity.pdf", 1.0,
             r"Sensitivity to the two quantities still awaiting "
             r"first-principles input: $\tau_c\propto U_{\rm cap}^{0.91}$ and "
             r"$\propto\Omega_{\rm rel}^{1.26}$, against the Varvenne--Curtin "
             r"analytic exponent of $4/3$. An earlier draft reported an "
             r"exponent of $0.20$ for $U_{\rm cap}$; that was an artefact of "
             r"the discontinuous core cutoff of Sec.~5.1."),
    "FIGS1": ("figS1_dd_map.pdf", 0.62,
              r"Differential displacement map of the screw dipole used for the "
              r"first-principles cells, verifying that two opposite cores were "
              r"created and that the slipped area lies between them. "
              r"Periodicity residual 0.03\% of $|\mathbf{b}|$."),
}


def main():
    figs = FIGS
    if ZH:
        from fig_captions_zh import FIGS_ZH
        figs = FIGS_ZH
    s = open("body_raw.tex").read()

    i = s.index(r"\begin{center}\rule")
    s = s[s.index("\n", i) + 1:]

    s = s.replace(r"\subsubsection{", "@@SUB@@{").replace(
                  r"\subsection{", "@@SEC@@{")
    s = s.replace("@@SEC@@{", r"\section{").replace("@@SUB@@{", r"\subsection{")

    def lvl(m):
        t = m.group(1)
        head = r"\subsection{" if re.match(r"^\d+\.\d+\s", t) else r"\section{"
        return head + t

    s = re.sub(r"\\section\{([^}]*)", lvl, s)
    s = re.sub(r"\\section\{\d+\.\s+", r"\\section{", s)
    s = re.sub(r"\\subsection\{\d+\.\d+\s+", r"\\subsection{", s)

    unnumbered = (["摘要", "数据可用性", "工具声明", "参考文献"] if ZH else
                  ["Abstract", "Data availability", "Tool disclosure",
                   "References"])
    for t in unnumbered:
        s = s.replace(r"\section{" + t, r"\section*{" + t)

    s = s.replace(r"\label{", r"\label{sec:")

    for k, (fn, w, cap) in figs.items():
        env = ("\\begin{figure}[htbp]\\centering\n"
               f"\\includegraphics[width={w}\\linewidth]{{{fn}}}\n"
               f"\\caption{{{cap}}}\n\\label{{fig:{k.lower()}}}\n"
               "\\end{figure}")
        s = s.replace(f"@@{k}@@", env)

    assert "@@" not in s, "an unreplaced figure marker remains"
    open(OUT, "w").write(s)
    print(OUT, "written")


if __name__ == "__main__":
    main()
