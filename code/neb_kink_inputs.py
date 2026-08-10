"""
VASP NEB inputs for the Peierls barrier of the a/2<111> screw dislocation in
bcc Ta, with and without a Lu atom in the core.

WHAT IS COMPUTED
    The screw glides between adjacent easy-core sites on a {110} plane.  The
    barrier along that path is the Peierls energy V_0 per unit length, which
    sets the kink energy through the sine-Gordon relation

        E_k = (2 h / pi) sqrt(2 Gamma V_0)

    so a chemical reduction of V_0 propagates to the kink-pair nucleation
    enthalpy as

        delta(2 E_k) / (2 E_k) = (1/2) delta V_0 / V_0
        kappa == 2 E_k^pure - 2 E_k^Lu ~ E_k * (delta V_0 / V_0)

    That is the quantity kink_kmc.py needs.

CELL DESIGN: BOTH CORES TRANSLATE TOGETHER
    The quadrupolar cell holds a dipole (+b, -b).  We translate BOTH cores by
    the same glide vector.  Then
      - the dipole vector d is unchanged, so the elastic interaction energy
        between the two dislocations is the same in every image and does not
        contaminate the barrier;
      - the net plastic strain is (b - b) * dx / A = 0, so the cell needs no
        retilting between images and every image shares one POSCAR cell.
    Translating only one core would change d and mix a spurious elastic term
    into the barrier.

    Consequence for extraction:
        pure cell   : E_barrier = 2 * V_0^pure * L_z
        Lu cell     : E_barrier = (V_0^pure + V_0^Lu) * L_z
        => V_0^Lu * L_z = E_barrier(Lu) - E_barrier(pure)/2

INITIAL PATH
    Images are built by placing the cores at intermediate glide positions and
    regenerating the anisotropic (Stroh) displacement field for each one, NOT
    by linear interpolation of Cartesian coordinates.  Linear interpolation of
    a dislocation field passes through unphysical configurations with atoms
    far off their columns and routinely makes NEB converge to the wrong path.

USAGE
    python neb_kink_inputs.py
"""

import os
import json
import numpy as np

from ase.io import write

import screw_dft_cells as S

ROOT = "neb_kink"
N_IMAGES = 5             # intermediate images (VASP IMAGES tag)


# ------------------------------------------------------------- glide vector
def glide_vector(perfect, Q, r1, a0=None, e_index=1):
    """
    Vector from one easy core to the NEXT EASY CORE in the {110} glide plane.

    Triangle centroids alternate in orientation: the nearest centroid to an
    easy core is the HARD core, at d_col/sqrt(3) = 1.56 A, and taking it would
    make the NEB path end on the saddle instead of in the next valley.  Easy
    cores of the same orientation form a triangular lattice with the same
    parameter as the atomic columns, d_col = a0*sqrt(6)/3 = 2.70 A, so we
    select the centroid at that distance which is best aligned with the glide
    direction e2.
    """
    a0 = a0 or S.A0
    d_col = a0*np.sqrt(6)/3

    P = perfect.get_positions() @ Q.T
    cols = np.unique(np.round(P[:, :2], 4), axis=0)
    R1 = (Q @ r1)[:2]

    cents = []
    for c in cols:
        d = np.linalg.norm(cols - c, axis=1)
        tri = cols[np.argsort(d)[:3]]
        cents.append(tri.mean(axis=0))
    cents = np.unique(np.round(np.array(cents), 3), axis=0)

    v = cents - R1
    r = np.linalg.norm(v, axis=1)
    same_sublattice = np.abs(r - d_col) < 0.20*d_col
    if not same_sublattice.any():
        raise RuntimeError("no easy core found at the expected spacing")
    score = np.where(same_sublattice, v[:, e_index], -1e9)
    j = int(np.argmax(score))

    g = np.zeros(3)
    g[:2] = v[j]
    return g, float(np.linalg.norm(g[:2]))


# ------------------------------------------------------------------- INCAR
INCAR_NEB = """SYSTEM = {name}
# ---------------- NEB ----------------
IMAGES = {nimg}        ! intermediate images only; endpoints go in 00 and {last}
SPRING = -5            ! negative = nudged elastic band with |k| = 5 eV/A^2.
                       ! For a dislocation path use a STIFF spring: the images
                       ! otherwise slide down into the two valleys and leave
                       ! the saddle unresolved.
LCLIMB = .TRUE.        ! climbing image: needed for the barrier itself, not
                       ! just the path.  Run a plain NEB first, then restart
                       ! with LCLIMB once the path is roughly converged.
ICHAIN = 0             ! standard NEB (VTST)
IOPT   = 3             ! quick-min via VTST tools; if using plain VASP set
                       ! IOPT = 0 and IBRION = 3, POTIM = 0
LNEBCELL = .FALSE.     ! cell is identical in all images by construction

# ---------------- ionic relaxation ----------------
IBRION = 3             ! damped MD; IBRION = 2 (CG) is NOT reliable with NEB
POTIM  = 0             ! step size handled by IOPT when VTST is available
NSW    = 400
EDIFFG = -0.01         ! eV/A.  Tighter than this is rarely reachable for a
                       ! 147-atom dislocation cell and is not needed: the
                       ! barrier converges long before the forces do.

# ---------------- electronic ----------------
ENCUT  = 500
PREC   = Accurate
LREAL  = .FALSE.
EDIFF  = 1E-6          ! one order looser than the dipole-tensor runs: NEB
                       ! needs forces, not stresses
ALGO   = Normal
NELM   = 120
ISPIN  = 1             ! Lu_3 PAW, 4f frozen in core
ISMEAR = 1
SIGMA  = 0.15

LWAVE  = .FALSE.
LCHARG = .FALSE.

# ---------------- parallelisation ----------------
! run with NCORE ~ sqrt(cores_per_image); total cores must be a multiple
! of (IMAGES) so each image gets its own group
"""

KPOINTS = """dense along the line, sparse in plane
0
Gamma
2 1 16
0 0 0
"""


def write_image(at, path):
    os.makedirs(path, exist_ok=True)
    write(os.path.join(path, "POSCAR"), at, format="vasp", direct=True,
          sort=True)


def write_run(name, images, nimg):
    root = os.path.join(ROOT, name)
    os.makedirs(root, exist_ok=True)
    for i, at in enumerate(images):
        write_image(at, os.path.join(root, f"{i:02d}"))
    with open(os.path.join(root, "INCAR"), "w") as f:
        f.write(INCAR_NEB.format(name=name, nimg=nimg,
                                 last=f"{len(images)-1:02d}"))
    with open(os.path.join(root, "KPOINTS"), "w") as f:
        f.write(KPOINTS)
    with open(os.path.join(root, "POTCAR.README"), "w") as f:
        f.write("cat <Lu_3>/POTCAR <Ta_pv>/POTCAR > POTCAR  "
                "(ASE sorts Lu before Ta)\n"
                "Pure-Ta runs need only Ta_pv.\n")
    return root


# ------------------------------------------------------------------- build
def build(mode, perfect, Q, r1, r2, b, g, nimg=N_IMAGES):
    """
    mode = 'pure' or 'Lu'.  Both cores translate by t*g, t = 0 .. 1.
    """
    images = []
    lu_index = None
    for i in range(nimg + 2):
        t = i/(nimg + 1)
        at, u = S.dipole_displacement(perfect, r1 + t*g, r2 + t*g, b, Q)
        if mode == "Lu":
            if lu_index is None:
                # nearest column to the core at t = 0, i.e. inside the
                # easy-core triangle the dislocation is sitting in
                P = perfect.get_positions() @ Q.T
                R1 = (Q @ r1)[:2]
                lu_index = int(np.argmin(np.linalg.norm(P[:, :2] - R1,
                                                        axis=1)))
            sym = at.get_chemical_symbols()
            sym[lu_index] = "Lu"
            at.set_chemical_symbols(sym)
        images.append(at)
    return images, lu_index


def main():
    a = S.A0
    b = np.sqrt(3)/2*a
    st, Q = S.screw_stroh()
    perfect = S.perfect_crystal(a)

    r1 = S.easy_core_position(perfect, Q, (0.25, 0.25))
    r2 = S.easy_core_position(perfect, Q, (0.75, 0.75))
    g_frame, gd = glide_vector(perfect, Q, r1)
    g = Q.T @ g_frame                        # back to the crystal frame

    print(f"cell: {len(perfect)} atoms, |b| = {b:.4f} A")
    print(f"glide vector: |g| = {gd:.4f} A "
          f"(a0*sqrt(6)/3 = {a*np.sqrt(6)/3:.4f} A expected)")
    if abs(gd - a*np.sqrt(6)/3) > 0.15:
        print("  WARNING: glide vector does not match the {110} easy-easy "
              "spacing; check the core search")

    manifest = []
    for mode in ["pure", "Lu"]:
        images, lu = build(mode, perfect, Q, r1, r2, b, g)
        root = write_run(f"neb_{mode}", images, N_IMAGES)
        # sanity: no image should have collapsed bonds
        dmin = min(_min_dist(im) for im in images)
        print(f"  {mode:>4}: {len(images)} images -> {root}   "
              f"min NN over all images = {dmin:.3f} A "
              f"({'OK' if dmin > 0.8*b else 'COLLAPSED'})")
        manifest.append(dict(mode=mode, path=root, n_images=len(images),
                             lu_index=lu, glide=gd))

    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(ROOT, "barriers.json"), "w") as f:
        json.dump({"pure": 0.0, "Lu": 0.0}, f, indent=2)

    print(f"\nfill {ROOT}/barriers.json with the climbing-image barriers "
          f"(eV per cell), then:")
    print("  V0_Lu * Lz = E_barrier(Lu) - E_barrier(pure)/2")
    print("  kappa ~ E_k * (V0_pure - V0_Lu)/V0_pure    -> kink_kmc.py")


def _min_dist(at):
    d = at.get_all_distances(mic=True)
    np.fill_diagonal(d, 1e9)
    return float(d.min())


if __name__ == "__main__":
    main()
