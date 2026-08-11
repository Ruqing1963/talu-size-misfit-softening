"""Write every reported numerical result to CSV.

Values are transcribed from the recorded simulation runs archived in this
repository; each file names the script that produced it so the number can be
regenerated.  Critical stresses are quoted to the precision justified by the
9.6 % realisation scatter (two significant figures).
"""
import csv, json, os
import numpy as np

D = "repo/data"

def w(name, header, rows, note):
    with open(os.path.join(D, name), "w", newline="") as f:
        f.write(f"# {note}\n")
        c = csv.writer(f); c.writerow(header); c.writerows(rows)
    print(f"  {name:34s} {len(rows):3d} rows")

# ---------------------------------------------------------------- elasticity
w("elastic_parameters.csv",
  ["quantity", "symbol", "value", "unit", "source"],
  [["lattice parameter", "a0", 3.3058, "A", "input"],
   ["Burgers vector", "b", 2.8629, "A", "a0*sqrt(3)/2"],
   ["elastic constant", "C11", 266.0, "GPa", "literature"],
   ["elastic constant", "C12", 158.2, "GPa", "literature"],
   ["elastic constant", "C44", 87.4, "GPa", "literature"],
   ["Zener anisotropy", "A_Z", 1.622, "-", "2C44/(C11-C12)"],
   ["bulk modulus", "K", 194.1, "GPa", "(C11+2C12)/3"],
   ["relaxation volume", "Omega_rel", 11.45, "A^3", "hard sphere, placeholder"],
   ["elastic dipole", "P", 13.87, "eV", "K*Omega_rel"],
   ["isotropic amplitude", "A_iso", 3.1955, "eV.A", "stroh_field.py"],
   ["edge line tension", "Gamma_edge", 0.940, "eV/A", "de Wit-Koehler"],
   ["screw line tension", "Gamma_screw", 3.935, "eV/A", "de Wit-Koehler"],
   ["naive line tension", "mu b^2/2", 1.765, "eV/A", "for comparison"],
   ["theoretical strength", "mu/30", 2.30, "GPa", "-"],
   ["Stroh validation error", "-", 2.6e-8, "-", "vs closed form"]],
  "Derived elastic parameters. stroh_field.py, depin3d.py")

w("screw_elastic_check.csv",
  ["case", "max_abs_eps_kk", "max_abs_U_meV", "probe_radius_A"],
  [["edge, anisotropic", 4.523e-2, 627.53, 5.0],
   ["screw, isotropic", 1.221e-15, 0.0000, 5.0],
   ["screw, anisotropic", 4.713e-3, 65.39, 5.0]],
  "Screw dilatation vanishes identically in isotropic elasticity. screw_check.py")

# ------------------------------------------------------------- edge depinning
w("edge_depinning_vs_ageing.csv",
  ["theta", "n_solute", "trough_depth_eV_per_A", "tau_c_GPa", "tau_c_sd_GPa",
   "tau_coherent_GPa"],
  [[0.00, 3477, -0.028, 0.155, 0.088, 0.099],
   [0.02, 3632, -0.050, 0.275, 0.049, 0.144],
   [0.05, 3532, -0.096, 0.411, 0.068, 0.225],
   [0.10, 3806, -0.179, 0.577, 0.047, 0.432],
   [0.25, 4099, -0.448, 1.248, 0.095, 1.102],
   [1.00, 5942, -1.896, 5.184, 0.166, 4.505]],
  "Nz=128 (L=366 A, rho~7e14 m^-2), 6 realisations, smooth core. depin3d.py")

w("varvenne_curtin_reference.csv",
  ["alpha", "basis", "tau_y0_GPa", "dE_b_eV"],
  [[0.0833, "VC default 1/12", 0.460, 0.607],
   [0.2662, "matched to Gamma_edge", 0.312, 0.894]],
  "VC is only comparable when alpha matches the line tension used; "
  "tau_y0 ~ alpha^(-1/3). depin3d.py")

# ------------------------------------------------------------- sensitivities
w("sensitivity_ucap.csv",
  ["field", "theta", "U_cap_eV", "tau_c_GPa", "tau_c_sd_GPa"],
  [["aniso", 0.00, 0.30, 0.812, 0.093], ["aniso", 0.00, 0.50, 1.220, 0.130],
   ["aniso", 0.00, 0.70, 1.148, 0.104], ["aniso", 0.00, 1.00, 0.914, 0.310],
   ["aniso", 0.00, 1.50, 1.278, 0.116],
   ["aniso", 0.25, 0.30, 0.385, 0.045], ["aniso", 0.25, 0.50, 0.906, 0.098],
   ["aniso", 0.25, 0.70, 1.234, 0.079], ["aniso", 0.25, 1.00, 1.399, 0.146],
   ["aniso", 0.25, 1.50, 1.788, 0.164],
   ["iso", 0.25, 0.30, 0.441, 0.054], ["iso", 0.25, 0.50, 1.010, 0.101],
   ["iso", 0.25, 0.70, 1.391, 0.133], ["iso", 0.25, 1.00, 1.579, 0.144],
   ["iso", 0.25, 1.50, 1.948, 0.092]],
  "tau_c ~ U_cap^0.91 (aniso, theta=0.25). Isotropic elasticity "
  "overestimates by 9-15 % at theta=0.25. sensitivity.py")

w("sensitivity_omega_rel.csv",
  ["Omega_rel_A3", "tau_c_GPa", "tau_c_sd_GPa"],
  [[7.00, 0.601, 0.040], [9.00, 0.882, 0.153], [11.45, 1.190, 0.060],
   [14.00, 1.423, 0.024], [17.00, 1.912, 0.077]],
  "tau_c ~ Omega_rel^1.26; VC analytic exponent is 4/3. omega_scan.py")

# ------------------------------------------------------------ solute-solute
w("atmosphere_solute_interaction.csv",
  ["omega_eV", "J_eV", "n_solute", "trough_depth_eV_per_A", "tau_c_GPa",
   "tau_c_sd_GPa", "enhancement_over_J0"],
  [[0.00, -0.000, 5824, -1.888, 5.197, 0.189, 1.00],
   [0.02, -0.020, 6494, -2.263, 6.440, 0.061, 1.24],
   [0.04, -0.040, 7774, -2.896, 7.602, 0.189, 1.46],
   [0.06, -0.060, 10669, -4.083, 9.392, 0.258, 1.81]],
  "Semi-grand-canonical, T_age=900 K. J=0 reproduces analytic Fermi-Dirac "
  "to 0.8 sigma. atmosphere3d.py")

w("ageing_threshold.csv",
  ["omega_eV", "theta", "tau_c_GPa", "theta_star_at_mu_over_30"],
  [[0.00, 0.10, 0.281, 0.574], [0.00, 0.25, 0.641, 0.574],
   [0.00, 0.40, 1.343, 0.574], [0.00, 0.50, 1.864, 0.574],
   [0.00, 0.55, 2.164, 0.574], [0.00, 0.65, 2.725, 0.574],
   [0.04, 0.10, 0.421, 0.413], [0.04, 0.25, 0.741, 0.413],
   [0.04, 0.50, 3.126, 0.413]],
  "Ageing fraction at which tau_c reaches mu/30 = 2.30 GPa. theta_shift.py")

# --------------------------------------------------------------- screw
w("screw_vs_edge.csv",
  ["theta", "tau_c_screw_GPa", "tau_c_edge_GPa", "ratio"],
  [[0.00, 0.0006, 0.155, 0.004], [0.25, 0.0006, 1.248, 0.0005],
   [1.00, 0.0006, 5.184, 0.0001]],
  "Screw values are upper bounds: depinning occurred at the first nonzero "
  "stress step of a 0.30 GPa ramp. screw_depin.py")

w("screw_landscape.csv",
  ["character", "max_binding_eV", "trough_depth_eV_per_A", "max_Fpin_eV_per_A",
   "Gamma_eV_per_A"],
  [["screw", -0.050, -0.0201, 0.287, 3.935],
   ["edge", -0.411, -1.5335, 1.163, 0.940]],
  "Screw trough is 76x shallower and its line 4.19x stiffer. screw_depin.py")

# ------------------------------------------------------------------ KMC
w("kmc_flow_stress.csv",
  ["T_K", "tau_pure_GPa", "tau_alloy_GPa", "ratio", "kink_density",
   "softening_term_meV"],
  [[100, 0.815, 1.008, 1.237, 0.0136, 84.6],
   [150, 0.235, 0.517, 2.198, 0.0125, 56.4],
   [200, 0.012, 0.187, 15.649, 0.0162, 42.3],
   [300, 0.0005, 0.100, float("nan"), 0.0132, 28.2],
   [450, float("nan"), 0.030, float("nan"), 0.0201, 18.8]],
  "c0=2 at.% Lu, U_cap=0.30 eV placeholder, sigma=0.0382 eV, 2E_k=0.90 eV. "
  "softening_term = sigma^2/(2 kT). NaN = outside the stress bracket. "
  "kink_kmc.py")

w("kappa_scan.csv",
  ["T_K", "kappa_eV", "tau_pure_GPa", "tau_alloy_GPa", "ratio"],
  [[150, 0.00, 0.234, 0.527, 2.251], [150, 0.05, 0.234, 0.367, 1.567],
   [150, 0.10, 0.234, 0.298, 1.272], [150, 0.15, 0.234, 0.266, 1.137],
   [150, 0.20, 0.234, 0.168, 0.717], [150, 0.30, 0.234, 0.161, 0.688],
   [250, 0.00, 0.001, 0.109, 79.834], [250, 0.05, 0.001, 0.118, 86.280],
   [250, 0.10, 0.001, 0.114, 82.821], [250, 0.15, 0.001, 0.110, 80.322],
   [250, 0.20, 0.001, 0.083, 60.861], [250, 0.30, 0.001, 0.020, 14.523]],
  "kappa* = 0.166 eV at 150 K = 18.5 % of 2E_k. No crossing at 250 K. "
  "kappa_scan.py")

w("kmc_parameters.csv",
  ["quantity", "symbol", "value", "unit", "provenance"],
  [["kink energy", "E_k", 0.45, "eV", "literature, pure Ta"],
   ["kink pair enthalpy", "2E_k", 0.90, "eV", "-"],
   ["kink migration", "E_km", 0.02, "eV", "literature"],
   ["kink-kink attraction", "E_kk", 0.61, "eV.segment", "mu b^2 h^2/[8pi(1-nu)a]"],
   ["Peierls stress", "tau_P", 0.90, "GPa", "DFT range; expt ~0.4"],
   ["core binding", "U_cap", 0.30, "eV", "PLACEHOLDER"],
   ["landscape sd", "sigma", 0.0382, "eV/segment", "measured"],
   ["attempt frequency", "nu_D", 1e13, "1/s", "assumed"],
   ["target velocity", "v", 3.49e4, "A/s", "gdot=1e-3, rho=1e12"]],
  "kink_kmc.py")

# -------------------------------------------------------------- convergence
cj = json.load(open("convergence.json"))
rows = []
for k in ["dx_grid", "n_tau", "R_keep", "Nz"]:
    for v, (m, sem, n) in sorted(cj[k].items(), key=lambda x: float(x[0])):
        rows.append([k, float(v), round(m, 4), round(sem, 4), n])
w("convergence.csv", ["parameter", "value", "tau_c_GPa", "sem_GPa", "n"],
  rows, "One-parameter sweeps at theta=0.25. convergence.py / convergence.json")

sv = np.array(cj["seeds"]["values"])
w("realisation_scatter.csv", ["realisation", "tau_c_GPa"],
  [[i+1, round(float(v), 4)] for i, v in enumerate(sv)],
  f"n={sv.size}, mean={sv.mean():.3f}, sd={sv.std(ddof=1):.3f} GPa "
  f"({100*sv.std(ddof=1)/sv.mean():.1f} %). Two significant figures justified.")

w("line_length_scaling.csv",
  ["Nz", "L_A", "rho_equiv_m-2", "tau_c_GPa", "sem_GPa"],
  [[32, 92, 1.2e17, 1.560, 0.067], [64, 183, 3.0e16, 1.379, 0.067],
   [128, 366, 7.5e15, 1.379, 0.064], [256, 733, 1.9e15, 1.299, 0.047],
   [512, 1466, 4.7e14, 1.189, 0.046], [1024, 2932, 1.2e14, 1.122, 0.034],
   [2048, 5863, 2.9e13, 1.101, 0.022]],
  "Weakest-link statistics: tau_c = 1.885 - 0.107 ln(Nz), rms 0.0345 GPa. "
  "No plateau. 6 realisations. convergence.py")

print("\ndata layer written to repo/data/")
