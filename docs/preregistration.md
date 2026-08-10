# Preregistration Protocol

## Elastic Size Misfit at Its Physical Ceiling: Does Statistical Disorder Cause Solid Solution Softening in bcc Ta? A Kink-Pair Kinetic Monte Carlo Study of Ta(Lu) with a Pre-specified First-Principles Test

**Status:** Preregistration. Simulations described in Sections 3.1–3.3 are complete; the density-functional calculations in Section 4 have **not** been run. No DFT total energies, stresses or barriers exist at the time of registration; all input decks are archived unchanged alongside this document.

**Authors:** Ronghua Wu (Zhuhai Jiuchongtian Aviation Technology Co., Ltd., Zhuhai, China; wrh@gdtzn.com); Ruqing Chen (GUT Geoservice Inc., Montreal, Canada; ruqing@hotmail.com)

**Version:** 1.0
**Registration date:** [to be filled on upload]
**Repository:** https://github.com/Ruqing1963/talu-size-misfit-softening

**Code archive DOI:** [to be filled on upload]

---

## 1. Background

Solid solution softening — the reduction of flow stress on alloying, observed in bcc metals such as W(Re), Mo(Re) and Fe(Si) below roughly 0.2 T_m — is generally attributed to a solute-induced change in kink-pair nucleation on ½⟨111⟩ screw dislocations. Two mechanisms are usually invoked without being separated quantitatively:

**(M1) Statistical.** A random solute field makes the kink-pair nucleation barrier a random variable. Nucleation samples the low tail of that distribution, so the effective activation enthalpy is reduced by σ²/2k_BT for a Gaussian barrier distribution of variance σ². Since this term grows as temperature falls, disorder alone can in principle soften a bcc metal at low temperature.

**(M2) Chemical/core.** The solute directly alters the dislocation core, lowering the Peierls barrier itself. This lies outside continuum elasticity and requires electronic-structure input.

The two are not distinguished by measuring softening; both predict it. They are distinguished by asking whether M1 is *sufficient*.

## 2. Rationale for Ta(Lu)

Ta(Lu) is not a metallurgically accessible alloy. Equilibrium solubility of Lu in Ta is below 0.1 at.%, no intermetallic phase exists, and the relevant experimental setting is a supersaturated film grown far from equilibrium. We adopt it deliberately, as an upper-bound probe, for one reason:

With r_Lu = 1.73 Å and r_Ta = 1.46 Å the linear misfit is 18.5% and the relaxation volume from atomic volumes is ΔV = 11.4 Å³ — at the ceiling attainable in a refractory bcc host.

The scaling argument makes this choice load-bearing rather than incidental:

- the M1 softening term scales as σ²/2k_BT, hence as **U²**, where U is the solute–dislocation interaction strength;
- the competing hardening channels (kink-migration drag; extreme-value bias of the nucleation saddle) scale as **U**.

The softening-to-hardening ratio therefore grows **linearly with misfit**. Ta(Lu) is the most favourable case M1 can present in a bcc host, so a negative result propagates to weaker solutes rather than being confined to this system. A positive result would not have this property, and we state in advance that a positive result would be reported as system-specific.

## 3. Completed simulations (reported as background, not tested here)

### 3.1 Elastic fields

Anisotropic Stroh sextic solution with Ta elastic constants C11/C12/C44 = 266.0/158.2/87.4 GPa (Zener A = 1.622). Validated against the closed-form isotropic result A sin θ/r to a maximum relative error of 2.6 × 10⁻⁸.

Line tensions from the Dewit–Koehler relation Γ = E + d²E/dθ²: Γ_edge = 0.940 eV/Å (the factor 1 − 2ν = 0.32 makes an edge dislocation 0.53× as stiff as the naive μb²/2 estimate) and Γ_screw = 3.935 eV/Å, a ratio of 4.19.

For the ½⟨111⟩ screw the dilatational coupling vanishes identically in isotropic elasticity (|ε_kk| < 1.2 × 10⁻¹⁵ numerically) and survives only because (111) is not a mirror plane of m3̄m. The surviving field is cos 3θ/r, verified to scale exactly as 1/r, to contain a third harmonic 80× larger than any other, and to be invariant under arbitrary rotation of the in-plane basis. Its amplitude is an order of magnitude below the edge case.

### 3.2 Edge-dislocation depinning

Quasi-static, drag-free depinning of a flexible line from a frozen atmosphere. Random solid solution: τ_c = 0.16 ± 0.09 GPa. Fully equilibrated atmosphere: τ_c = 5.2 ± 0.2 GPa, above the theoretical shear strength μ/30 = 2.30 GPa, so a completely aged atmosphere admits no breakaway. Semi-grand-canonical sampling with solute–solute interaction raises τ_c by 1.2–1.8× over the Bernoulli approximation and moves the crossing of μ/30 from an ageing fraction θ* = 0.57 to 0.41.

For the screw, τ_c^screw/τ_c^edge < 10⁻³ at every ageing fraction: the atmosphere does not pin the dislocation that controls low-temperature yield.

### 3.3 Kink-pair kinetic Monte Carlo

Solid-on-solid reduction of the continuum functional, with kink–kink elastic attraction E_int(w) = E_kk/w, rejection-free BKL over three event classes. Across 100–450 K a random Lu field raises the flow stress at every temperature. The fluctuation-induced barrier reduction reaches 85 meV at 100 K, or 9.4% of the pure nucleation enthalpy 2E_k = 0.90 eV, and is outweighed by kink-migration drag and by the extreme-value bias of the saddle.

Introducing a phenomenological chemical softening κ reverses the sign above

> **κ*(U_cap = 0.30 eV) = 0.17 ± 0.03 eV at 150 K**, or 19% of the pure kink-pair enthalpy.

Above ~250 K no value of κ produces softening, since pure Ta is already athermal at the imposed strain rate.

### 3.4 Frozen numerical parameters

These are fixed by the convergence study archived with the code and will not be altered after DFT results arrive:

| Parameter | Value | Basis |
|---|---|---|
| dx_grid | 0.20 Å | converged to 4 significant figures |
| n_τ | 500 over 8 GPa | 0.6% |
| R_keep | 60 Å | 0.3% |
| N_z | 128 (L = 366 Å, ρ ≈ 7 × 10¹⁴ m⁻²) | see below |
| realisations | ≥ 4 per point | 9.6% scatter → ±5% |
| strain rate | γ̇ = 10⁻³ s⁻¹, ρ_m = 10¹² m⁻² | v_target = 3.49 × 10⁴ Å/s |
| test temperature for κ* | 150 K | only window where pure Ta is not athermal |

Depinning from a random landscape has no thermodynamic limit: τ_c = 1.885 − 0.107 ln N_z (rms 0.035 GPa) over N_z = 32–2048, with no plateau. Critical stresses are therefore reported for a stated segment length, never as intensive constants.

## 4. Data collection: 39 first-principles calculations

None of these have been run. Input decks are archived unchanged.

### Set A — elastic dipole tensor and interchange energy (12 cells)

Generated by `dft_dipole_cells.py`. Fixed cell, ions only (ISIF = 2), ENCUT = 500 eV, LREAL = .FALSE., PREC = Accurate, Ta_pv + Lu_3 PAW, k-point density held constant across sizes.

| # | Case | Atoms | Yields |
|---|---|---|---|
| 1–8 | bulk_n{2,3,4,5}, Lu_n{2,3,4,5} | 16, 54, 128, 250 | P_ij = −V(σ_defect − σ_bulk); Ω_rel by 1/N extrapolation |
| 9–12 | pair_s{1,2,3,4} | 128 | ω_n = −[E(2Lu@n) + E_pure − 2E(1Lu)]; ω_eff = (Σ z_n ω_n)/6 |

Built-in checks: Ω_rel must be positive (Lu is oversized; a negative value indicates a stress sign-convention error and the parse script raises); ω_eff must be positive (Ta–Lu immiscibility; a negative value contradicts the phase diagram).

### Set B — screw core binding energy (13 cells)

Generated by `screw_dft_cells.py`. 7 × 7 × 1 quadrupolar dipole cell, 147 atoms, anisotropic Stroh displacement field, periodicity restored by the numerically measured slip offset δ_n (residual 0.03% of |b|), k-mesh 2 × 1 × 16.

| # | Case | Yields |
|---|---|---|
| 13 | perfect | reference |
| 14 | screw | reference |
| 15–24 | Lu at 10 symmetry-inequivalent columns, r = 1.54–6.77 Å | E_b(r, θ) |
| 25 | Lu far reference, r = 28.1 Å | subtraction reference |

E_b(site) = E(screw + Lu@site) − E(screw + Lu@far), same cell. Angular sampling is deliberate: the pairs (4.08 Å, 19.1°)/(4.09 Å, 100.9°) and (5.54 Å, 46.4°)/(5.56 Å, 73.8°) sit at equal radius and different azimuth and therefore measure the cos 3θ amplitude directly.

Defines **U_cap^DFT ≡ max_site |E_b|**.

### Set C — nudged elastic band (14 image calculations)

Generated by `neb_kink_inputs.py`. Two NEB runs of 7 images each (5 intermediate + 2 endpoints), IMAGES = 5, SPRING = −5, IBRION = 3, EDIFFG = −0.01 eV/Å, plain NEB first then restart with LCLIMB.

| # | Run | Yields |
|---|---|---|
| 26–32 | neb_pure | E_b^pure |
| 33–39 | neb_Lu (Lu on a core column) | E_b^Lu |

Both cores translate by the same glide vector g = 2.6995 Å (= a₀√6/3, easy core to next easy core, not to the intervening hard core). The dipole separation and hence the elastic interaction energy are unchanged, and the net plastic strain is (b − b)Δx/A = 0, so all images share one cell and no spurious elastic work enters the barrier. Images are built by regenerating the Stroh field at intermediate core positions, not by linear interpolation of coordinates.

Extraction:

    V₀^pure L_z = E_b^pure / 2
    V₀^Lu   L_z = E_b^Lu − E_b^pure / 2

## 5. Pre-specified decision rule

### 5.1 Definitions

From Set B:

    U_cap^DFT = max_site |E_b(site)|

From Set C, with E_k = (2h/π)√(2ΓV₀) the sine-Gordon kink energy:

    κ_DFT = 2(E_k^pure − E_k^Lu) = 2 E_k^pure [ 1 − √(V₀^Lu / V₀^pure) ]

The threshold is recomputed, not reused:

    κ*(U_cap^DFT) = the value of κ at which τ_y^alloy / τ_y^pure = 1 at T = 150 K,
                    obtained by rerunning kappa_scan.py with U_cap := U_cap^DFT
                    and every other parameter as frozen in Section 3.4.

This step is mandatory. κ* = 0.17 eV is defined relative to the placeholder U_cap = 0.30 eV and must not be quoted independently of it. The fluctuation term scales as U_cap², so a factor-of-two revision changes κ* substantially.

### 5.2 The rule

> **If κ_DFT < κ*(U_cap^DFT), we will publicly report that the chemical softening hypothesis (M2) is falsified for Ta(Lu) within this model and parameter space.**
>
> **If κ_DFT ≥ κ*(U_cap^DFT), we will report M2 as supported for this system**, subject to the system-specificity caveat in Section 2.

Both outcomes will be reported with equal prominence, in this preprint, regardless of which occurs.

### 5.3 Conditions that render the test inconclusive

Declared in advance so that they cannot be invoked selectively:

1. NEB fails to converge to EDIFFG = −0.01 eV/Å within 400 ionic steps in either run, or the climbing image does not sit at a maximum along the path.
2. Set B yields U_cap^DFT such that the KMC stress bisection saturates at either bracket bound at 150 K, so κ* is not resolvable.
3. Either built-in sign check in Set A fails (Ω_rel ≤ 0 or ω_eff ≤ 0), indicating an error in convention or in the reference structure.
4. The core-corrected interaction Δ(r,θ) = E_b^DFT − U_elastic does not decay to within the DFT noise floor by r ≈ 3b, indicating that the 147-atom cell is too small for the elastic subtraction to be meaningful.

In any of these cases we will report the calculation as inconclusive and state which condition was triggered, rather than adjusting the rule.

### 5.4 Alternatives the test does **not** discriminate

The rule tests M2 against M1 only. It does not address, and a falsification of M2 would not exclude:

- solute modification of the **kink migration** barrier E_km rather than the nucleation barrier;
- solute modification of the local **line tension** Γ;
- non-Schmid effects and the {112} twinning/antitwinning asymmetry of bcc screws, absent from this model;
- softening arising from interstitial impurity scavenging, a mechanism proposed for W(Re) that has no analogue in our solute-only model.

These are stated as out of scope, not as excluded.

## 6. Analyses declared exploratory

Any result not derivable from Sections 4–5 will be labelled exploratory. In particular: the angular fit of Δ(r,θ) to cos(3θ + φ); the effect of ω_eff on the edge atmosphere; the SOC correction described below; and any re-analysis at N_z ≠ 128.

## 7. Additional methodological commitments

**4f treatment.** Lu is 4f¹⁴ with the closed shell roughly 5 eV below E_F. A Hubbard correction acts only to shift filled states rigidly and cannot alter forces or barriers; LDA+U will not be used. The 4f contribution will instead be bounded by comparing barriers computed with f frozen in the core (Lu_3) against an explicit-f potential, and the difference reported as an upper bound.

**Spin–orbit coupling.** SOC is first-order inactive for J = 0 Lu but significant for Ta 5d. The path will be converged scalar-relativistically; SOC will be applied as single-point corrections at the endpoints and the climbing image, with ISYM = 0. SOC-corrected barriers are exploratory and will not be substituted into the decision rule.

## 8. Known limitations carried into the test

1. The SOS kinetics take the kink-pair barrier as a maximum over nucleus widths, so disorder enters through an extreme-value statistic biased toward higher barriers. This is a property of the discretisation, not the physics, and it systematically favours hardening. Our result is therefore a **conservative bound on softening**, not a symmetric estimate.
2. E_k = 0.45 eV and τ_P = 0.90 GPa are literature estimates for pure Ta, not values derived for this system. τ_P carries a known discrepancy: density-functional values near 1 GPa exceed experimental extrapolations of roughly 0.4 GPa, a disagreement common to bcc transition metals and unresolved here.
3. The SOS lattice maps the solute field onto a simple-cubic neighbour network (z = 6) rather than the bcc first-neighbour shell (z = 8), rescaling the effective interchange energy without changing the topology of the argument.
4. Ω_rel = 11.4 Å³ is presently a hard-sphere estimate from atomic volumes and will be replaced by the Set A value; τ_c scales as Ω_rel^1.26.

## 9. Code and data availability

All generation and analysis scripts, the convergence study, and every unmodified DFT input deck are archived with this registration. The output templates (`stresses.json`, `energies.json`, `barriers.json`) are archived containing zeros, demonstrating that no results existed at registration.

Both extraction pipelines have been validated by synthetic round-trip tests: injected values of Ω_rel and ω_eff are recovered exactly by the parsers.

## 10. Authorship and tool disclosure

Large language model assistance (Anthropic Claude) was used for derivation checking, code development and manuscript drafting. Consistent with Zenodo and ICMJE guidance, the model is not listed as an author: it cannot take responsibility for the work or consent to publication. All physical claims, parameter choices and the decision rule above are the responsibility of the human authors, who have verified the derivations independently.

Several errors were identified and corrected during development and are documented in the code comments rather than silently removed, including: a 57% grid artefact from a discontinuous core cutoff; a periodicity violation of 10% of |b| from an incorrect dipole tilt formula; branch-cut accumulation in the image sum of the screw displacement field; and a linear stress bisection that saturated silently at both brackets. Two sensitivity conclusions reported in earlier drafts were reversed by these corrections and are not carried forward.
