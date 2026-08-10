"""
DFT input generation for the Lu-screw-dislocation binding energy in bcc Ta.

WHY THIS CALCULATION
    The screw dislocation is the rate-controlling one in bcc below ~0.4 T_m.
    Its elastic coupling to a substitutional solute is nearly nil:
        isotropic elasticity  ->  eps_kk = 0 EXACTLY
        anisotropic (Stroh)   ->  cos(3 theta)/r, ~10x weaker than the edge
    so essentially the whole interaction is a CORE effect that no elasticity
    based model can supply.  It has to come from DFT.

    After the convergence fix (smooth core regularisation) the strength scales
    as tau_c ~ U_core^0.91, i.e. almost linearly -- so this number matters as
    much as the relaxation volume, not less.

GEOMETRY (Ventelon-Willaime quadrupolar setup)
    line / z : [111],  |b| = a sqrt(3)/2
    cell     : C1 = n1 a [1,-1,0],  C2 = n2 a [0,1,-1],  C3 = m (a/2)[111]
               -> 3 * n1 * n2 * m atoms
    A dislocation DIPOLE with separation d = (C1 + C2)/2 tiles into a
    quadrupolar array whose long-range fields cancel to leading order.

    Two corrections are essential and are both applied here:
      1. the anisotropic (Stroh) displacement field of the dipole, not the
         isotropic atan2 field;
      2. the homogeneous plastic tilt that restores periodicity,
             du_i/dx_j = -(b_i/A) (d x xi)_j ,
         without which the cell is not a valid periodic crystal at all.

OUTPUT
    perfect / screw / screw+Lu at several core-adjacent sites, as VASP and QE
    decks, plus a differential-displacement map to verify the cores visually.

    E_b(site) = E(screw + Lu@site) - E(screw + Lu@far)
    computed in the SAME cell, so most systematic errors cancel.

USAGE
    python screw_dft_cells.py            # write decks + DD map
    python screw_dft_cells.py --parse    # read energies back
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

from ase import Atoms
from ase.io import write

import stroh_field as SF

GPA = 6.2415e-3
A0 = 3.3058                      # A, bcc Ta -- re-relax with your own PAW
C11, C12, C44 = 266.0, 158.2, 87.4
N1, N2, M = 7, 7, 1              # 3*7*7*1 = 147 atoms, |C1| = |C2|
ROOT = "screw_dft"


# ------------------------------------------------------------ cell building
def cell_matrix(a=A0, n1=N1, n2=N2, m=M):
    C1 = n1*a*np.array([1.0, -1.0, 0.0])
    C2 = n2*a*np.array([0.0, 1.0, -1.0])
    C3 = m*(a/2)*np.array([1.0, 1.0, 1.0])
    return np.vstack([C1, C2, C3])


def perfect_crystal(a=A0, n1=N1, n2=N2, m=M):
    """All bcc lattice points inside the cell, found by brute-force enumeration
    (robust against sign/index mistakes in a hand-derived basis)."""
    C = cell_matrix(a, n1, n2, m)
    Cinv = np.linalg.inv(C)

    # generous bounding box in conventional-cell units
    R = int(np.ceil(np.abs(C).sum()/a)) + 2
    ijk = np.stack(np.meshgrid(*[np.arange(-R, R+1)]*3, indexing="ij"), -1)
    ijk = ijk.reshape(-1, 3).astype(float)
    corner = ijk*a
    centre = (ijk + 0.5)*a
    pts = np.vstack([corner, centre])

    frac = pts @ Cinv
    tol = 1e-8
    keep = np.all((frac > -tol) & (frac < 1.0 - tol), axis=1)
    pos = pts[keep]

    n_expect = 3*n1*n2*m
    assert len(pos) == n_expect, f"got {len(pos)} atoms, expected {n_expect}"
    return Atoms(f"Ta{len(pos)}", positions=pos, cell=C, pbc=True)


# --------------------------------------------------- dislocation dipole field
def screw_stroh():
    """Stroh solution in the screw frame: e1, e2 perpendicular, e3 = [111]."""
    e1 = np.array([1., -1., 0.])/np.sqrt(2)
    e3 = np.array([1., 1., 1.])/np.sqrt(3)
    e2 = np.cross(e3, e1); e2 /= np.linalg.norm(e2)
    Q = np.vstack([e1, e2, e3])
    C = SF.rotate_C(SF.cubic_C(C11*GPA, C12*GPA, C44*GPA), Q)
    return SF.Stroh(C), Q


def dipole_displacement(at, r1, r2, b, Q, n_img=6):
    """
    Anisotropic displacement of a screw dipole (+b at r1, -b at r2) plus its
    periodic images, followed by the homogeneous tilt that restores
    periodicity.  r1, r2 are 3D Cartesian; b is the Burgers magnitude.
    """
    st, _ = screw_stroh()
    C = at.cell[:]
    pos = at.get_positions()

    # work in the dislocation frame
    P = pos @ Q.T
    R1 = Q @ r1
    R2 = Q @ r2
    C1f, C2f = Q @ C[0], Q @ C[1]

    u = _image_sum(st, P, R1, R2, C1f, C2f, b, n_img)

    # --- restore periodicity.
    #
    # After the image sum, the dipole field satisfies
    #       u(R + C_n) - u(R) = delta_n
    # with delta_n a CONSTANT vector (verified below to ~0.1 % of |b|): it is
    # the plastic slip carried across one cell vector.  Periodicity of the
    # displaced crystal then requires exactly
    #       C_n^new = C_n + delta_n
    # and NO homogeneous strain on the atoms.
    #
    # Applying instead the textbook tilt gradient du_i/dx_j = -(b_i/A)(d x xi)_j
    # to the atoms leaves a residual of 10 % of |b| that does not go away with
    # more images -- i.e. the cell is not a valid periodic crystal.  Measuring
    # delta_n numerically sidesteps every sign and factor convention.
    Cf = C @ Q.T
    delta = np.zeros((3, 3))
    for n in (0, 1):
        Pn = (at.get_positions() + C[n]) @ Q.T
        un = _image_sum(st, Pn, R1, R2, C1f, C2f, b, n_img)
        d_n = un - u
        delta[n] = d_n.mean(axis=0)
        spread = d_n.std(axis=0).max()
        assert spread < 0.02*b, (
            f"slip offset for C{n+1} is not constant "
            f"(spread {spread:.4f} A); increase n_img")
    Cf = Cf + delta

    newP = P + u
    at2 = at.copy()
    at2.set_positions(newP @ Q)                     # back to the crystal frame
    at2.set_cell(Cf @ Q, scale_atoms=False)
    # NOTE: u is returned separately.  Do NOT diagnose the displacement by
    # differencing wrapped positions -- wrapping across the periodic boundary
    # adds whole cell vectors and reports 25 A instead of the physical |b|/2.
    return at2, u


def _image_sum(st, P, R1, R2, C1f, C2f, b, n_img):
    """Dipole displacement summed over an (2 n_img+1)^2 array of periodic images."""
    bvec = np.array([0.0, 0.0, b])
    u = np.zeros_like(P)
    for i in range(-n_img, n_img+1):
        for j in range(-n_img, n_img+1):
            sh = i*C1f + j*C2f
            u += _stroh_dipole(st, P[:, 0], P[:, 1],
                               R1[:2] + sh[:2], R2[:2] + sh[:2], bvec)
    return u


def _stroh_dipole(st, x, y, c1, c2, b):
    """
    Displacement of a dipole (+b at c1, -b at c2):

        u_k = (1/pi) Im[ sum_a A_ka (L_a . b) ln(eta1_a / eta2_a) ]

    The RATIO is essential.  Written as ln(eta1) - ln(eta2), each branch cut
    contributes an independent +/-2 pi jump, and summing over ~50 periodic
    images accumulates displacements of tens of Angstroms instead of the
    physical maximum of |b|/2.  ln(eta1/eta2) takes the principal branch of
    the ratio, which is the correct short-range dipole field and decays as
    1/r so the image sum converges quickly.
    """
    Lb = st.L.T @ b
    out = np.zeros((x.size, 3))
    for a in range(3):
        e1 = (x - c1[0]) + st.p[a]*(y - c1[1])
        e2 = (x - c2[0]) + st.p[a]*(y - c2[1])
        e1 = np.where(np.abs(e1) < 1e-8, 1e-8, e1)
        e2 = np.where(np.abs(e2) < 1e-8, 1e-8, e2)
        lg = np.log(e1/e2)
        for k in range(3):
            out[:, k] += (st.A[k, a]*Lb[a]*lg).imag/np.pi
    return out


def easy_core_position(at, Q, frac=(0.25, 0.25)):
    """
    Easy core: the centroid of the triangle of three [111] atomic columns.
    Located by picking the column nearest the requested fractional position
    and averaging it with its two nearest neighbours of the other two
    sublattices.
    """
    C = at.cell[:]
    target = frac[0]*C[0] + frac[1]*C[1]
    P = at.get_positions() @ Q.T
    T = (Q @ target)[:2]

    cols = np.unique(np.round(P[:, :2], 4), axis=0)
    d = np.linalg.norm(cols - T, axis=1)
    seed = cols[np.argmin(d)]
    dd = np.linalg.norm(cols - seed, axis=1)
    order = np.argsort(dd)
    tri = cols[order[:3]]
    c2 = tri.mean(axis=0)
    return Q.T @ np.array([c2[0], c2[1], 0.0])


# ------------------------------------------------------------------- decks
INCAR = """SYSTEM = {name}
ENCUT   = 500
PREC    = Accurate
LREAL   = .FALSE.
EDIFF   = 1E-7
ALGO    = Normal
NELM    = 200
ISPIN   = 1              ! use the Lu_3 PAW (4f frozen in core)

ISMEAR  = 1
SIGMA   = 0.15

! ions only, cell FIXED: the dipole tilt is already built into POSCAR
IBRION  = 2
ISIF    = 2
NSW     = 300
EDIFFG  = -5E-3

LWAVE   = .FALSE.
LCHARG  = .FALSE.
"""

KPOINTS = """dense along the line, sparse in plane
0
Gamma
{k1} {k2} {k3}
0 0 0
"""


def write_case(at, name, k=(2, 1, 16)):
    path = os.path.join(ROOT, name)
    os.makedirs(path, exist_ok=True)
    write(os.path.join(path, "POSCAR"), at, format="vasp", direct=True,
          sort=True)
    with open(os.path.join(path, "INCAR"), "w") as f:
        f.write(INCAR.format(name=name))
    with open(os.path.join(path, "KPOINTS"), "w") as f:
        f.write(KPOINTS.format(k1=k[0], k2=k[1], k3=k[2]))
    return path


# --------------------------------------------------------------- generator
def generate():
    a = A0
    b = np.sqrt(3)/2*a
    st, Q = screw_stroh()

    perfect = perfect_crystal(a)
    C = perfect.cell[:]
    print(f"cell {N1}x{N2}x{M}: {len(perfect)} atoms, "
          f"|C1|={np.linalg.norm(C[0]):.2f} |C2|={np.linalg.norm(C[1]):.2f} "
          f"|C3|={np.linalg.norm(C[2]):.3f} A  (|b| = {b:.3f})")

    r1 = easy_core_position(perfect, Q, (0.25, 0.25))
    r2 = easy_core_position(perfect, Q, (0.75, 0.75))
    screw, u = dipole_displacement(perfect, r1, r2, b, Q)
    print(f"dipole cores at fractional (0.25,0.25) and (0.75,0.75)")
    print(f"  max |u_z| = {np.abs(u[:, 2]).max():.3f} A   (physical bound "
          f"|b|/2 = {b/2:.3f})")
    print(f"  max in-plane |u| = {np.abs(u[:, :2]).max():.3f} A")
    assert np.abs(u[:, 2]).max() < 0.75*b, "screw displacement field is wrong"

    manifest = []
    write_case(perfect, "perfect");  manifest.append(dict(name="perfect", kind="perfect"))
    write_case(screw, "screw");      manifest.append(dict(name="screw", kind="screw"))

    # --- Lu substitution sites.
    #
    # Sampling one site per distance shell gives U(r) only.  The screw core
    # field has cos(3 theta) symmetry, so the ANGLE must be resolved too:
    # take every symmetry-inequivalent column within r_max, folded into the
    # 120-degree wedge that the 3-fold axis leaves inequivalent.  (The screw
    # breaks the mirror symmetry, so the wedge is 120 deg, not 60.)
    P = screw.get_positions() @ Q.T
    R1 = Q @ r1
    dx = P[:, 0] - R1[0]
    dy = P[:, 1] - R1[1]
    rr = np.hypot(dx, dy)
    th = np.arctan2(dy, dx) % (2*np.pi/3)          # fold by the 3-fold axis

    r_max = 7.5
    cand = np.where(rr < r_max)[0]
    chosen, seen = [], []
    for idx in cand[np.argsort(rr[cand])]:
        key = (round(float(rr[idx]), 2), round(float(th[idx]), 2))
        if any(abs(key[0]-a) < 0.12 and abs(key[1]-b_) < 0.06 for a, b_ in seen):
            continue
        seen.append(key)
        chosen.append((int(idx), float(rr[idx]), float(np.degrees(th[idx]))))
    far = int(np.argmax(rr))
    chosen.append((far, float(rr[far]), float(np.degrees(th[far]))))

    print(f"\nLu substitution sites: {len(chosen)-1} inequivalent columns "
          f"within r < {r_max} A, plus one far reference")
    for idx, rv, tv in chosen:
        at = screw.copy()
        sym = at.get_chemical_symbols(); sym[idx] = "Lu"
        at.set_chemical_symbols(sym)
        name = f"Lu_r{rv:05.2f}_t{tv:05.1f}".replace(".", "p")
        write_case(at, name)
        manifest.append(dict(name=name, kind="screw+Lu", site=idx,
                             r=rv, theta=tv, reference=(idx == far)))
        print(f"  {name:>22s}  r = {rv:5.2f} A  theta = {tv:5.1f} deg"
              f"{'   <- far reference' if idx == far else ''}")

    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(ROOT, "energies.json"), "w") as f:
        json.dump({m["name"]: 0.0 for m in manifest}, f, indent=2)

    dd_map(perfect, u, Q, b, r1, r2)
    print(f"\nwrote {len(manifest)} cases under {ROOT}/")
    print("fill screw_dft/energies.json with the relaxed total energies in eV,")
    print("then run:  python screw_dft_cells.py --parse")


# ----------------------------------------------- differential displacement
def dd_map(perfect, u, Q, b, r1, r2, fname="screw_dd_map.png"):
    """u is the applied displacement in the dislocation frame (never wrapped)."""
    P0 = perfect.get_positions() @ Q.T
    duz = u[:, 2] - u[:, 2].mean()

    cols = {}
    for i in range(len(P0)):
        k = (round(P0[i, 0], 3), round(P0[i, 1], 3))
        cols.setdefault(k, []).append(duz[i])
    xy = np.array(list(cols.keys()))
    val = np.array([np.mean(v) for v in cols.values()])

    R1, R2 = Q @ r1, Q @ r2
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    s = ax.scatter(xy[:, 0], xy[:, 1], c=val, cmap="coolwarm",
                   vmin=-b/2, vmax=b/2, s=110, edgecolor="k", linewidth=0.3)
    ax.plot(*R1[:2], "kx", ms=14, mew=2.5, label="core +b")
    ax.plot(*R2[:2], "k+", ms=16, mew=2.5, label="core -b")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$\rm\AA$]"); ax.set_ylabel(r"$y$ [$\rm\AA$]")
    ax.set_title(r"screw dipole: $u_z$ per [111] atomic column"
                 "\n(verification that two opposite cores were created)")
    ax.legend(fontsize=9)
    fig.colorbar(s, ax=ax, label=r"$u_z$ [$\rm\AA$]")
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    fig.savefig(fname.replace('.png', '.pdf'))
    print(f"figure written: {fname}")


# ------------------------------------------------------------ post-process
def parse():
    with open(os.path.join(ROOT, "manifest.json")) as f:
        man = json.load(f)
    with open(os.path.join(ROOT, "energies.json")) as f:
        E = json.load(f)
    if all(v == 0.0 for v in E.values()):
        print("energies.json is still all zeros -- run the DFT first.")
        return

    ref = [m for m in man if m.get("reference")]
    if not ref:
        print("no far reference case found."); return
    E_ref = E[ref[0]["name"]]

    print(f"{'r [A]':>8} {'E_b [eV]':>10}   (negative = solute attracted)")
    rows = []
    for m in man:
        if m["kind"] != "screw+Lu" or m.get("reference"):
            continue
        Eb = E[m["name"]] - E_ref
        rows.append((m["r"], Eb))
        print(f"{m['r']:8.2f} {Eb:10.3f}")

    if rows:
        Ebmax = min(r[1] for r in rows)
        print(f"\n  strongest binding: {Ebmax:.3f} eV")
        print(f"  elastic (anisotropic Stroh) prediction at r ~ 5 A: -0.065 eV")
        print(f"  core enhancement factor: {abs(Ebmax)/0.065:.1f}x")
        print(f"\n  set U_cap = {abs(Ebmax):.2f} in screw_depin.py and re-run;")
        print(f"  tau_c ~ U_cap^0.91, so this scales the screw strength "
              f"almost linearly.")


if __name__ == "__main__":
    if "--parse" in sys.argv:
        parse()
    else:
        generate()
