<!-- https://github.com/Ruqing1963/talu-size-misfit-softening -->

# Elastic Size Misfit at Its Physical Ceiling: a Ta(Lu) Probe for Solid Solution Softening in bcc Metals

**Status: preregistration.** Elasticity and kinetic Monte Carlo simulations are complete. **No density-functional calculations have been run.** The three input tarballs in `03_dft_neb_templates/` contain unmodified VASP/QE decks; their output templates (`stresses.json`, `energies.json`, `barriers.json`) are archived containing zeros, which is the evidence that no results existed at registration.

---

## ⚠ Read this before quoting κ* = 0.17 eV

The chemical-softening threshold **κ\* = 0.17 ± 0.03 eV (150 K)** is *not* a standalone result. It is defined relative to a placeholder screw-core binding energy **U_cap = 0.30 eV**, which is a literature-range guess pending the DFT in `03_dft_neb_templates/screw_dft_cells.py`.

The fluctuation-softening term scales as U_cap², so a factor-of-two revision of U_cap changes κ\* substantially. Section 5.1 of the protocol therefore requires κ\* to be **recomputed**, not reused, once U_cap^DFT is available. Quoting 0.17 eV without U_cap = 0.30 eV attached is a misreading.

---

## What this is

A multiscale pipeline asking whether *statistical* elastic disorder is sufficient to produce solid solution softening in a bcc metal, or whether direct core chemistry is required.

Ta(Lu) is not a metallurgically accessible alloy: equilibrium solubility below 0.1 at.%, no intermetallic phase. It is used deliberately as an **upper-bound probe**. Its 18.5 % linear misfit and 11.4 Å³ relaxation volume put the elastic size interaction at the ceiling attainable in a refractory host. Because the softening contribution scales as U² while the competing hardening channels scale as U, this is the most favourable case elastic disorder can present — so a *negative* result propagates to weaker solutes. A positive result would not, and is declared system-specific in advance.

**Headline finding (background, not the preregistered test):** across 100–450 K a random Lu field *raises* the flow stress of the rate-controlling ½⟨111⟩ screw at every temperature, despite a fluctuation-induced barrier reduction reaching 9.4 % of the pure nucleation enthalpy.

**Preregistered test:** if the DFT-measured core barrier reduction κ_DFT < κ\*(U_cap^DFT), the chemical-softening hypothesis is falsified for this system within this model. Both outcomes will be reported.

## Repository layout

```
code/     00_atmosphere/     2D Kawasaki MC; 3D semi-grand-canonical sampling
          01_elastic/        Stroh sextic solution; screw dilatation test
          02_kmc/            edge depinning, sensitivities, kink-pair KMC, kappa scan
          03_dft_templates/  generators for the 39 pre-specified DFT calculations
          04_convergence/    convergence study and its recorded results
          build_data.py      writes data/*.csv
          make_figures.py    builds figures/*.pdf from data/*.csv
data/     16 CSV files: every reported number, each naming its source script
figures/  publication figures as vector PDF (plus PNG previews)
paper/    manuscript.md / manuscript_zh.md, references.bib, LaTeX sources,
          compiled main.pdf (English) and main_zh.pdf (Chinese)
          build.sh / build_zh.sh rebuild each end to end (pandoc + xelatex)
dft_inputs/  three tarballs of unmodified VASP/QE decks
docs/     preregistration protocol
```

### Rebuilding

```bash
python code/build_data.py     # CSV layer, seconds
python code/make_figures.py   # vector figures from CSV, seconds
cd paper && ./build.sh          # English, 16 pages
cd paper && ./build_zh.sh       # Chinese, 15 pages (needs texlive-lang-chinese
                                # and a Noto Serif CJK SC font)
```

**References: 18 of 18 verified.** Verification caught nine errors, documented
in the header of `paper/references.bib`: two wrong titles, a wrong journal
series, a wrong issue number, a wrong page range, three wrong DOIs, a wrong ISBN
check digit, and one monograph mis-typed as a journal article. A wrong DOI is
not cosmetic — preprint servers resolve DOIs automatically, creating a permanent
link to the wrong paper. Two independently supplied bibliographies, each
described as complete or fully verified, contained these errors between them;
one was also present in an earlier draft of our own file.

Figures are generated from the CSV layer rather than from live simulation, so
the paper rebuilds in seconds and every plotted point is traceable to a data
row. Map-type figures (field maps, the differential-displacement map, the 2D
atmosphere) come from their own scripts, which now emit PDF alongside PNG.

## Reproducing

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/code/00_atmosphere:$PWD/code/01_elastic:$PWD/code/02_kmc:$PWD/code/03_dft_templates
python code/01_elastic/stroh_field.py       # validates to 2.6e-8, seconds
python code/04_convergence/convergence.py   # ~20 min
python code/02_kmc/kink_kmc.py              # ~15 min
python code/02_kmc/kappa_scan.py            # ~25 min
```

Scripts import each other by module name, so `PYTHONPATH` must cover the code
subdirectories, or the tree can be flattened into one working directory.

**Reproducibility caveat.** Numba kernels use `fastmath=True` and Numba's internal RNG, whose stream is independent of NumPy's and is not guaranteed stable across Numba versions. Seeds are recorded, but exact bitwise reproduction requires the pinned versions in `requirements.txt`. Reported quantities carry a 9.6 % realisation scatter and are quoted to two significant figures; this dominates any version-to-version drift.

## Known limitations

Summarised here, stated formally in §8 of the protocol:

1. The SOS kinetics take the kink-pair barrier as a maximum over nucleus widths, so disorder enters through an extreme-value statistic biased toward higher barriers. This is a discretisation artefact that favours hardening, making the result a **conservative bound on softening** rather than a symmetric estimate.
2. `U_cap` (0.30 eV) is a placeholder; κ\* is relative to it.
3. E_k = 0.45 eV and τ_P = 0.90 GPa are literature values for pure Ta, and τ_P carries a known DFT-vs-experiment discrepancy (~1 GPa vs ~0.4 GPa) common to bcc transition metals.
4. The SOS lattice uses a simple-cubic neighbour network (z = 6), not the bcc first-neighbour shell (z = 8).
5. τ_c has no thermodynamic limit: it falls as 1.885 − 0.107 ln N_z with no plateau to N_z = 2048. All critical stresses are reported for a stated segment length (default N_z = 128, L = 366 Å, ρ ≈ 7 × 10¹⁴ m⁻²), never as intensive constants.

## Corrections made during development

Documented in code comments rather than removed, because two sensitivity conclusions in earlier drafts were reversed by them:

- a 57 % grid artefact from a discontinuous core cutoff, fixed by smooth regularisation `U = A r sinθ/(r² + r_c²)`;
- a periodicity violation of 10 % of |b| from an incorrect dipole tilt formula, fixed by measuring the slip offset numerically (residual 0.03 %);
- branch-cut accumulation in the image sum of the screw displacement field, giving 25 Å displacements against a physical bound of |b|/2;
- a linear stress bisection that saturated silently at both brackets, making an alloy/pure ratio meaningless.

## Licence

Code: MIT (`LICENSE`). Documentation and figures: CC BY 4.0 (`LICENSE-docs`).

## Authors

- **Ronghua Wu** — Zhuhai Jiuchongtian Aviation Technology Co., Ltd., Zhuhai, China · wrh@gdtzn.com
- **Ruqing Chen** — GUT Geoservice Inc., Montreal, Canada · ruqing@hotmail.com

## Tool disclosure

A large language model (Anthropic Claude) was used for derivation checking, code development and manuscript drafting. Consistent with Zenodo and ICMJE guidance it is not an author: it cannot take responsibility for the work or consent to publication. All physical claims, parameter choices and the decision rule are the responsibility of the human author, who has verified the derivations independently.
