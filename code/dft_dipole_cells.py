"""
DFT input generation and post-processing for the elastic dipole tensor P_ij
of substitutional Lu in bcc Ta.

WHY THIS CALCULATION AND NOT THE DISLOCATION ONE
    Numerical sensitivity study (omega_scan.py, sensitivity.py) gives

        tau_c ~ Omega_rel^1.20        (Varvenne-Curtin analytic: ^1.33)
        tau_c ~ U_cap^0.20

    The relaxation volume is ~6x more important than the core binding energy.
    Omega_rel comes from a 128-atom BULK supercell (hours on a small cluster);
    the core binding energy needs a ~1000-atom dislocation dipole cell (weeks).
    The cheap calculation is the one that matters.  Do it first.

WHAT IS COMPUTED
    P_ij = -V ( sigma_ij[Lu in Ta] - sigma_ij[pure Ta] )     at FIXED cell
    Omega_rel = Tr(P) / (3K),   K = (C11 + 2 C12)/3
    By O_h site symmetry P_ij = P delta_ij; the off-diagonal terms are a
    numerical error estimate, not physics.

  optional second-order term (set STRAINS below):
    alpha_ijkl = -dP_ij/d eps_kl   -- the diaelastic polarisability, which is
    what couples a substitutional solute to a SCREW dislocation.

USAGE
    python dft_dipole_cells.py            # write all input decks
    python dft_dipole_cells.py --parse    # read stresses back, get P and Omega
"""

import os
import sys
import json
import numpy as np

from ase import Atoms
from ase.build import bulk
from ase.io import write

# ---------------------------------------------------------------- settings
A0_TA = 3.3058          # A, DFT-relaxed bcc Ta lattice constant -- RE-RELAX IT
                        # with your own functional/PAW before using these cells
SIZES = [2, 3, 4, 5]    # conventional-cell repeats -> 16, 54, 128, 250 atoms
STRAINS = [0.0]         # add e.g. [0.0, 0.005] to also get alpha_ijkl
ROOT = "dft_dipole"

C11, C12, C44 = 266.0, 158.2, 87.4      # GPa, experiment; replace with DFT
K_BULK = (C11 + 2*C12)/3.0              # GPa
GPA_TO_EV_A3 = 6.2415e-3


# ------------------------------------------------------------ cell building
def bcc_supercell(n, a0=A0_TA, solute=None):
    """n x n x n repeats of the CONVENTIONAL bcc cell (2 atoms), 2n^3 atoms."""
    at = bulk("Ta", "bcc", a=a0, cubic=True).repeat((n, n, n))
    if solute is not None:
        sym = at.get_chemical_symbols()
        sym[0] = solute                  # substitute at the origin
        at.set_chemical_symbols(sym)
    return at


def apply_strain(at, eps):
    """Uniform strain tensor eps (3x3) applied to cell and positions."""
    F = np.eye(3) + eps
    at = at.copy()
    at.set_cell(at.cell[:] @ F.T, scale_atoms=True)
    return at


# ---------------------------------------------- Lu-Lu pair interaction cells
# bcc neighbour shells, in units of the conventional lattice parameter
BCC_SHELLS = {
    1: (np.array([0.5, 0.5, 0.5]),  8, np.sqrt(3)/2),
    2: (np.array([1.0, 0.0, 0.0]),  6, 1.0),
    3: (np.array([1.0, 1.0, 0.0]), 12, np.sqrt(2)),
    4: (np.array([1.5, 0.5, 0.5]), 24, np.sqrt(11)/2),
}
PAIR_N = 4            # supercell repeats for the pair cells (4 -> 128 atoms)
PAIR_SHELLS = [1, 2, 3, 4]


def bcc_pair(shell, n=PAIR_N, a0=A0_TA):
    """
    Two Lu atoms as nth neighbours in a bcc Ta supercell.

    The lattice-gas interchange energy per bond in shell n is

        omega_n = -[ E(2Lu @ n) + E(pure) - 2 E(1 Lu) ]

    with the sign convention omega = 2 eps_AB - eps_AA - eps_BB, so that
    omega > 0 means unlike bonds are unfavourable, i.e. clustering, which is
    what J = -omega < 0 encodes in atmosphere3d.py.
    """
    at = bcc_supercell(n, a0)
    pos = at.get_positions()
    v, _, _ = BCC_SHELLS[shell]
    target = pos[0] + v*a0

    frac = np.linalg.solve(at.cell[:].T, (pos - target).T).T
    frac -= np.round(frac)                       # minimum image
    d = np.linalg.norm(frac @ at.cell[:], axis=1)
    j = int(np.argmin(d))
    assert d[j] < 1e-6 and j != 0, \
        f"no atom at shell {shell}; supercell too small?"

    sym = at.get_chemical_symbols()
    sym[0] = "Lu"
    sym[j] = "Lu"
    at.set_chemical_symbols(sym)
    return at


# --------------------------------------------------------------- VASP decks
INCAR = """SYSTEM = {name}
# --- electronic
ENCUT   = 500          ! well above ENMAX; stress needs a high cutoff (Pulay)
PREC    = Accurate
LREAL   = .FALSE.      ! reciprocal-space projectors: required for good stress
EDIFF   = 1E-7
ALGO    = Normal
NELM    = 200
ISPIN   = 1            ! Lu is 4f14, nonmagnetic; use the Lu_3 PAW (f in core)

# --- Brillouin zone (metal)
ISMEAR  = 1
SIGMA   = 0.15         ! check that T*S/atom < 1 meV in OUTCAR

# --- relaxation: IONS ONLY, CELL FIXED.  This is what makes sigma the
#     defect-induced residual stress from which P_ij is read off.
IBRION  = 2
ISIF    = 2
NSW     = 200
EDIFFG  = -1E-3        ! tight forces; stress converges slower than energy

# --- output
LWAVE   = .FALSE.
LCHARG  = .FALSE.
LORBIT  = 0
"""

KPOINTS = """Gamma-centred, density fixed across cell sizes
0
Gamma
{k} {k} {k}
0 0 0
"""


def kmesh_for(n, base_n=2, base_k=16):
    """Keep the k-point density fixed as the cell grows."""
    return max(2, int(round(base_k/n*base_n/2))*1 or 2)


def write_vasp_case(at, path, name, k):
    os.makedirs(path, exist_ok=True)
    write(os.path.join(path, "POSCAR"), at, format="vasp", direct=True,
          sort=True)
    with open(os.path.join(path, "INCAR"), "w") as f:
        f.write(INCAR.format(name=name))
    with open(os.path.join(path, "KPOINTS"), "w") as f:
        f.write(KPOINTS.format(k=k))
    with open(os.path.join(path, "POTCAR.README"), "w") as f:
        f.write("cat <Ta_pv>/POTCAR <Lu_3>/POTCAR > POTCAR\n"
                "Order must match the species order in POSCAR (ASE sorts "
                "alphabetically: Lu before Ta).\n"
                "Use Ta_pv (5p semicore) and Lu_3 (4f frozen in core).\n")


# ------------------------------------------------------------------ QE deck
QE = """&CONTROL
  calculation = 'relax'
  prefix = '{name}'
  pseudo_dir = './pseudo'
  tstress = .true.
  tprnfor = .true.
  forc_conv_thr = 1.0d-4
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = 60
  ecutrho = 600
  occupations = 'smearing'
  smearing = 'mp'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
/
&IONS
  ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
{species}
CELL_PARAMETERS angstrom
{cell}
ATOMIC_POSITIONS crystal
{pos}
K_POINTS automatic
{k} {k} {k} 0 0 0
"""


def write_qe_case(at, path, name, k):
    os.makedirs(path, exist_ok=True)
    syms = sorted(set(at.get_chemical_symbols()))
    mass = {"Ta": 180.948, "Lu": 174.967}
    species = "\n".join(f"  {s} {mass[s]} {s}.upf" for s in syms)
    cell = "\n".join("  " + " ".join(f"{v:.10f}" for v in row)
                     for row in at.cell[:])
    sp = at.get_scaled_positions()
    pos = "\n".join(f"  {s} " + " ".join(f"{v:.10f}" for v in sp[i])
                    for i, s in enumerate(at.get_chemical_symbols()))
    with open(os.path.join(path, f"{name}.in"), "w") as f:
        f.write(QE.format(name=name, nat=len(at), ntyp=len(syms),
                          species=species, cell=cell, pos=pos, k=k))


# ---------------------------------------------------------------- generator
def generate():
    manifest = []
    for n in SIZES:
        k = kmesh_for(n)
        for eps_mag in STRAINS:
            eps = np.zeros((3, 3))
            tag = ""
            if eps_mag != 0.0:
                eps[0, 0] = eps_mag           # uniaxial probe for alpha_11kl
                tag = f"_e{eps_mag:g}"
            for kind, solute in [("bulk", None), ("Lu", "Lu")]:
                at = bcc_supercell(n, solute=solute)
                if eps_mag != 0.0:
                    at = apply_strain(at, eps)
                name = f"{kind}_n{n}{tag}"
                path = os.path.join(ROOT, name)
                write_vasp_case(at, path, name, k)
                write_qe_case(at, path, name, k)
                manifest.append(dict(name=name, n=n, nat=len(at), kind=kind,
                                     kmesh=k, strain=eps_mag,
                                     volume=float(at.get_volume()),
                                     path=path))
                print(f"  {name:>16s}  {len(at):4d} atoms  "
                      f"k={k}x{k}x{k}  V={at.get_volume():9.2f} A^3")

    # --- Lu-Lu pair cells: the interchange energy omega, free of extra work
    #     because they reuse the bulk_n4 and Lu_n4 references above.
    print()
    k = kmesh_for(PAIR_N)
    for sh in PAIR_SHELLS:
        at = bcc_pair(sh)
        _, z, dfac = BCC_SHELLS[sh]
        name = f"pair_s{sh}"
        path = os.path.join(ROOT, name)
        write_vasp_case(at, path, name, k)
        write_qe_case(at, path, name, k)
        manifest.append(dict(name=name, n=PAIR_N, nat=len(at), kind="pair",
                             shell=sh, z=z, dist=float(dfac*A0_TA),
                             kmesh=k, path=path))
        print(f"  {name:>16s}  {len(at):4d} atoms  shell {sh}: "
              f"z={z:2d}, d={dfac*A0_TA:5.3f} A")

    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(ROOT, "stresses.json"), "w") as f:
        json.dump({m["name"]: [[0.0]*3]*3 for m in manifest
                   if m["kind"] != "pair"}, f, indent=2)
    with open(os.path.join(ROOT, "energies.json"), "w") as f:
        json.dump({m["name"]: 0.0 for m in manifest
                   if m["kind"] == "pair" or m["n"] == PAIR_N}, f, indent=2)

    print(f"\nwrote {len(manifest)} cases under {ROOT}/")
    print(f"  {ROOT}/stresses.json  <- Cauchy stress tensors in GPa "
          f"(tension positive)  -> Omega_rel")
    print(f"  {ROOT}/energies.json  <- relaxed total energies in eV "
          f"-> omega")
    print("then run:  python dft_dipole_cells.py --parse")


# ------------------------------------------------------------- post-process
def parse():
    with open(os.path.join(ROOT, "manifest.json")) as f:
        man_list = json.load(f)
    man = {m["name"]: m for m in man_list}
    with open(os.path.join(ROOT, "stresses.json")) as f:
        sig = {k: np.array(v, float) for k, v in json.load(f).items()}

    if all(np.allclose(v, 0) for v in sig.values()):
        print("stresses.json is still all zeros -- run the DFT first.")
        return

    rows = []
    for n in SIZES:
        b, d = f"bulk_n{n}", f"Lu_n{n}"
        if b not in sig or d not in sig:
            continue
        V = man[d]["volume"]                          # A^3
        # P_ij = -V * (sigma_defect - sigma_bulk), both at the SAME cell and
        # ENCUT so the Pulay contribution cancels in the difference.
        Pij = -V*(sig[d] - sig[b])*GPA_TO_EV_A3       # eV
        P = np.trace(Pij)/3.0
        aniso = np.abs(Pij - P*np.eye(3)).max()
        Om = np.trace(Pij)/(3*K_BULK*GPA_TO_EV_A3)    # A^3
        rows.append((man[d]["nat"], P, aniso, Om))
        print(f"  N={man[d]['nat']:4d}  P = {P:8.3f} eV   "
              f"Omega_rel = {Om:7.3f} A^3   "
              f"|off-diagonal| = {aniso:.4f} eV")

    if len(rows) < 2:
        print("need at least two cell sizes to extrapolate.")
        return

    # image-dipole interaction falls off as 1/V, i.e. linearly in 1/N
    N = np.array([r[0] for r in rows], float)
    Om = np.array([r[3] for r in rows], float)
    c = np.polyfit(1.0/N, Om, 1)
    Om_inf = c[1]
    print(f"\n  1/N extrapolation:  Omega_rel(N -> inf) = {Om_inf:.3f} A^3")
    print(f"  hard-sphere estimate from atomic volumes = 11.45 A^3")

    P_inf = Om_inf*K_BULK*GPA_TO_EV_A3
    print(f"  => P = K * Omega_rel = {P_inf:.3f} eV")

    # SIGN CHECK.  Lu is oversized in Ta; anything else means the stress sign
    # convention of your code is opposite to the one assumed here.
    if Om_inf <= 0:
        print("\n  *** SIGN ERROR: Lu is oversized in Ta, so Omega_rel MUST be")
        print("      positive.  Flip the sign of the stresses in stresses.json")
        print("      (VASP 'in kB' vs Cauchy tension-positive) and re-run.")
    else:
        d = 100*(Om_inf/11.45 - 1)
        print(f"\n  sign check passed.  DFT differs from the hard-sphere "
              f"estimate by {d:+.1f} %,")
        print(f"  which propagates to tau_c as "
              f"{100*((Om_inf/11.45)**1.26 - 1):+.1f} % (exponent 1.26).")
        print(f"\n  set  Omega_rel = {Om_inf:.3f}  in stroh_field.py and "
              f"sensitivity.py")

    parse_omega(man_list)


def parse_omega(man):
    """omega_n = -[E(2Lu@n) + E(pure) - 2 E(1Lu)], all in the same cell."""
    path = os.path.join(ROOT, "energies.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        E = json.load(f)
    need = [f"bulk_n{PAIR_N}", f"Lu_n{PAIR_N}"] + \
           [f"pair_s{s}" for s in PAIR_SHELLS]
    if any(E.get(k, 0.0) == 0.0 for k in need):
        print("\n  (energies.json still empty -- omega not evaluated)")
        return

    E_pure = E[f"bulk_n{PAIR_N}"]
    E_one = E[f"Lu_n{PAIR_N}"]
    print("\n" + "-"*58)
    print("  Lu-Lu interchange energy")
    print(f"  {'shell':>6} {'z':>3} {'d [A]':>7} {'omega_n [eV]':>13}")
    tot = 0.0
    for m in man:
        if m["kind"] != "pair":
            continue
        om = -(E[m["name"]] + E_pure - 2*E_one)
        tot += m["z"]*om
        print(f"  {m['shell']:6d} {m['z']:3d} {m['dist']:7.3f} {om:13.4f}")

    # map onto the single-J, z=6 lattice-gas model by preserving the
    # mean-field mixing enthalpy  sum_n z_n omega_n
    om_eff = tot/6.0
    print(f"\n  sum_n z_n omega_n = {tot:.4f} eV")
    print(f"  effective omega for the z=6 model = {om_eff:.4f} eV")
    if om_eff > 0:
        print(f"  omega > 0: Lu-Lu clustering, consistent with Ta-Lu "
              f"immiscibility")
    else:
        print(f"  omega < 0: ordering tendency -- would contradict the "
              f"experimental phase diagram; check the sign convention")
    Tc = 4.511*(abs(om_eff)/4)/8.617333e-5
    print(f"  bulk demixing Tc (3D Ising) = {Tc:.0f} K")
    print(f"\n  set omegas = [{om_eff:.3f}] in atmosphere3d.py; tau_c rises "
          f"by ~1.2-1.8x over the J=0 result across omega = 0.02-0.06 eV")


if __name__ == "__main__":
    if "--parse" in sys.argv:
        parse()
    else:
        generate()
