"""
Solute-dislocation interaction energy from the ELASTIC DIPOLE TENSOR, using
anisotropic elasticity (Stroh sextic formalism).

Replaces the isotropic  U = A sin(theta)/r  placeholder in depin3d.py with

        U(r, theta) = - P_ij eps_ij^disl(r, theta)

which is exact to first order in the solute strain field, for any crystal
anisotropy, and which takes P_ij directly from DFT.

SYMMETRY NOTE (this corrects an earlier claim)
    A substitutional solute on a bcc lattice site has the full O_h site point
    group.  A symmetric second-rank tensor invariant under O_h must be
    isotropic, so
                        P_ij = P delta_ij       exactly.
    There is NO first-order tetragonal / shear coupling for a substitutional
    solute in a cubic host, however large the size misfit.  Shear coupling
    enters only at second order through the diaelastic polarisability,
        U^(2) = -1/2 alpha_ijkl eps_ij eps_kl ,
    which is what couples a substitutional solute to a SCREW dislocation
    (whose field is pure shear).  For the edge dislocation treated here the
    first-order dilatational term dominates.

    P is fixed by the relaxation volume:   P = K * Omega_rel,  K = (C11+2C12)/3.

Coordinate frame for the bcc 1/2<111>{110} edge dislocation:
    e1 = [111]/sqrt(3)    glide direction  (b)
    e2 = [1-10]/sqrt(2)   glide plane normal
    e3 = [11-2]/sqrt(6)   line direction
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

GPA = 6.2415e-3          # 1 GPa in eV/Angstrom^3


# ------------------------------------------------------------ elastic tensor
def cubic_C(C11, C12, C44):
    """4-index stiffness tensor of a cubic crystal, in the crystal frame."""
    C = np.zeros((3, 3, 3, 3))
    d = np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = (
                        C12 * d[i, j] * d[k, l]
                        + C44 * (d[i, k] * d[j, l] + d[i, l] * d[j, k])
                    )
    for i in range(3):
        C[i, i, i, i] = C11
    return C


def isotropic_C(mu, nu):
    """Isotropic stiffness written in Voigt-cubic form (C11, C12, C44)."""
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    return cubic_C(lam + 2.0 * mu, lam, mu)


def rotate_C(C, Q):
    """C'_ijkl = Q_ia Q_jb Q_kc Q_ld C_abcd ; rows of Q are the new basis."""
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", Q, Q, Q, Q, C, optimize=True)


def dislocation_frame():
    """Orthonormal right-handed frame (e1, e2, e3) for 1/2<111>{110} edge."""
    e1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)      # b direction
    e2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)     # glide plane normal
    e3 = np.cross(e1, e2)                            # line direction
    e3 /= np.linalg.norm(e3)
    return np.vstack([e1, e2, e3])


# -------------------------------------------------------------------- Stroh
class Stroh:
    """
    Stroh sextic solution for a straight dislocation along x3.

    Solves  [Q + p(R + R^T) + p^2 T] A = 0 ,  L = (R^T + p T) A
    via the equivalent 6x6 linear eigenproblem, keeping the three roots with
    Im(p) > 0 and normalising  2 A_a . L_a = 1.
    """

    def __init__(self, Crot):
        Q = Crot[:, 0, :, 0]
        R = Crot[:, 0, :, 1]
        T = Crot[:, 1, :, 1]
        Ti = np.linalg.inv(T)

        N = np.zeros((6, 6))
        N[:3, :3] = -Ti @ R.T
        N[:3, 3:] = Ti
        N[3:, :3] = R @ Ti @ R.T - Q
        N[3:, 3:] = -R @ Ti

        w, v = np.linalg.eig(N)
        sel = np.argsort(-w.imag)[:3]          # the three roots with Im p > 0
        self.p = w[sel]
        A = v[:3, sel]
        L = v[3:, sel]

        for a in range(3):                      # normalise 2 A.L = 1
            s = 2.0 * (A[:, a] @ L[:, a])
            A[:, a] /= np.sqrt(s)
            L[:, a] /= np.sqrt(s)
        self.A, self.L = A, L

        # residual of the sextic, as a self-check
        self.residual = max(
            np.abs((Q + self.p[a] * (R + R.T) + self.p[a] ** 2 * T)
                   @ self.A[:, a]).max() for a in range(3))

    def distortion(self, x1, x2, b):
        """
        du_k/dx_m at (x1, x2) for Burgers vector b (in the dislocation frame).
        Returns an array of shape (..., 3, 3): [..., k, m].
        """
        x1 = np.asarray(x1, float)
        x2 = np.asarray(x2, float)
        Lb = self.L.T @ b                       # (3,)
        out = np.zeros(x1.shape + (3, 3))
        for a in range(3):
            eta = x1 + self.p[a] * x2
            f = Lb[a] / eta
            for k in range(3):
                out[..., k, 0] += (self.A[k, a] * f).imag / np.pi
                out[..., k, 1] += (self.A[k, a] * self.p[a] * f).imag / np.pi
        return out

    def dilatation(self, x1, x2, b):
        """eps_kk = du1/dx1 + du2/dx2  (du3/dx3 = 0 for a straight line)."""
        d = self.distortion(x1, x2, b)
        return d[..., 0, 0] + d[..., 1, 1]


# ------------------------------------------------- interaction energy field
def field_amplitude(stroh, bvec, r_probe=20.0, n=720):
    """The 1/r coefficient of eps_kk, i.e. max over theta of |r * eps_kk|."""
    th = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    ekk = stroh.dilatation(r_probe*np.cos(th), r_probe*np.sin(th), bvec)
    return float(np.abs(ekk).max()*r_probe)


def interaction_field(X, Y, P_scalar, stroh, bvec, U_cap, r_c=None):
    """
    U = -P_ij eps_ij = -P * eps_kk, with a SMOOTH core regularisation:

        U = -P eps_kk * r^2/(r^2 + r_c^2),     r_c = A_eff/(2 U_cap)

    where A_eff = P * (1/r coefficient of eps_kk).  This reduces to the raw
    field for r >> r_c, goes smoothly to zero at the core, and peaks at
    exactly U_cap -- so U_cap keeps its meaning as the core binding energy
    for whichever dislocation character is being modelled.

    A hard cutoff (U set to +/-U_cap inside r_core) instead puts a step into
    U(x).  Its numerical derivative is a spike of height ~1/dx and width ~dx,
    which made tau_c drift by 57 % with the grid spacing and was not even
    monotonic.  See convergence.py.

    P_scalar in eV, X/Y in Angstrom, returns eV.
    """
    if r_c is None:
        r_c = P_scalar*field_amplitude(stroh, bvec)/(2.0*U_cap)
    R2 = X*X + Y*Y
    U = np.zeros_like(R2)
    m = R2 > 0.0
    U[m] = (-P_scalar*stroh.dilatation(X[m], Y[m], bvec)
            * R2[m]/(R2[m] + r_c*r_c))
    return U


# ------------------------------------------------------------------- driver
def main():
    # --- Ta: Featherston & Neighbours, 300 K
    C11, C12, C44 = 266.0, 158.2, 87.4          # GPa
    a0 = 3.306                                   # A
    b = np.sqrt(3) / 2 * a0                      # 1/2<111>
    A_zener = 2 * C44 / (C11 - C12)
    K = (C11 + 2 * C12) / 3.0                    # GPa
    Omega_rel = 11.45                            # A^3, Lu in Ta (to be from DFT)
    P_scalar = K * GPA * Omega_rel               # eV

    print(f"Ta:  C11={C11}  C12={C12}  C44={C44} GPa")
    print(f"Zener anisotropy A = 2C44/(C11-C12) = {A_zener:.3f}  "
          f"(1.0 = isotropic)")
    print(f"K = {K:.1f} GPa,  Omega_rel = {Omega_rel} A^3  =>  "
          f"P = K*Omega_rel = {P_scalar:.2f} eV")
    print(f"b = {b:.4f} A\n")

    Qrot = dislocation_frame()
    bvec = np.array([b, 0.0, 0.0])               # b along e1 in the disl. frame

    # --- anisotropic
    C_aniso = rotate_C(cubic_C(C11 * GPA, C12 * GPA, C44 * GPA), Qrot)
    st_a = Stroh(C_aniso)
    print(f"anisotropic Stroh roots p = {np.round(st_a.p, 4)}")
    print(f"  sextic residual = {st_a.residual:.2e}")

    # --- isotropic reference with the SAME K (so only anisotropy differs)
    #     Voigt shear of Ta: mu_V = (C11-C12+3C44)/5
    mu_V = (C11 - C12 + 3 * C44) / 5.0
    nu_V = (3 * K - 2 * mu_V) / (2 * (3 * K + mu_V))
    C_iso = rotate_C(isotropic_C(mu_V * GPA, nu_V), Qrot)
    st_i = Stroh(C_iso)
    print(f"isotropic reference: mu_V = {mu_V:.1f} GPa, nu_V = {nu_V:.3f}")
    print(f"  sextic residual = {st_i.residual:.2e}\n")

    # --- closed-form isotropic check:  U = A_iso sin(theta)/r
    A_iso = (1 + nu_V) * (mu_V * GPA) * b * Omega_rel / (3 * np.pi * (1 - nu_V))
    th = np.linspace(0.01, 2 * np.pi - 0.01, 400)
    r_t = 10.0
    U_closed = A_iso * np.sin(th) / r_t
    U_stroh_i = -P_scalar * st_i.dilatation(r_t * np.cos(th), r_t * np.sin(th),
                                            bvec)
    err = np.abs(U_stroh_i - U_closed).max() / np.abs(U_closed).max()
    print(f"VALIDATION  isotropic Stroh vs closed form  A sin(th)/r :")
    print(f"  A_iso = {A_iso:.4f} eV*A,   max relative error = {err:.2e}\n")

    U_stroh_a = -P_scalar * st_a.dilatation(r_t * np.cos(th), r_t * np.sin(th),
                                            bvec)
    amp_i = np.abs(U_stroh_i).max()
    amp_a = np.abs(U_stroh_a).max()
    print(f"at r = {r_t} A:")
    print(f"  isotropic   max|U| = {amp_i*1000:7.2f} meV  at theta = "
          f"{np.degrees(th[np.argmax(np.abs(U_stroh_i))]):.1f} deg")
    print(f"  anisotropic max|U| = {amp_a*1000:7.2f} meV  at theta = "
          f"{np.degrees(th[np.argmax(np.abs(U_stroh_a))]):.1f} deg")
    print(f"  anisotropy changes the binding amplitude by "
          f"{100*(amp_a/amp_i - 1):+.1f} %")

    # --- 2D maps
    g = np.linspace(-30, 30, 481)
    X, Y = np.meshgrid(g, g, indexing="ij")
    Ua = interaction_field(X, Y, P_scalar, st_a, bvec, 0.70)
    Ui = interaction_field(X, Y, P_scalar, st_i, bvec, 0.70)

    np.savez("u_field_aniso.npz", x=g, y=g, U=Ua, P=P_scalar,
             p_roots=st_a.p, A=st_a.A, L=st_a.L, b=b,
             C=(C11, C12, C44), Omega_rel=Omega_rel)
    print("\ntabulated field written: u_field_aniso.npz")

    make_figure(th, U_stroh_i, U_stroh_a, U_closed, g, Ua, Ui, r_t, err)


def make_figure(th, Ui, Ua, Uc, g, Ua2d, Ui2d, r_t, err):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].plot(np.degrees(th), Uc * 1000, "k-", lw=3, alpha=0.35,
               label=r"closed form $A\sin\theta/r$")
    ax[0].plot(np.degrees(th), Ui * 1000, "b--", lw=1.6,
               label="Stroh, isotropic")
    ax[0].plot(np.degrees(th), Ua * 1000, "r-", lw=2,
               label="Stroh, anisotropic Ta")
    ax[0].set_xlabel(r"$\theta$ [deg]"); ax[0].set_ylabel(r"$U$ [meV]")
    ax[0].set_title(rf"(a) angular dependence at $r$={r_t:.0f} $\rm\AA$"
                    "\n" rf"isotropic validation: rel. err = {err:.1e}")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[0].axhline(0, color="0.5", lw=0.7)

    v = 0.35
    im = ax[1].imshow(Ua2d.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                      extent=[g[0], g[-1], g[0], g[-1]])
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xlabel(r"$x \parallel [111]$ [$\rm\AA$]")
    ax[1].set_ylabel(r"$y \parallel [1\bar{1}0]$ [$\rm\AA$]")
    ax[1].set_title(r"(b) $U(x,y)$, anisotropic Ta [eV]")
    fig.colorbar(im, ax=ax[1], shrink=0.85)

    ratio = np.where(np.abs(Ui2d) > 5e-3, Ua2d / np.where(Ui2d == 0, 1, Ui2d),
                     np.nan)
    im = ax[2].imshow(ratio.T, origin="lower", cmap="PuOr", vmin=0.7, vmax=1.3,
                      extent=[g[0], g[-1], g[0], g[-1]])
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].set_xlabel(r"$x$ [$\rm\AA$]"); ax[2].set_ylabel(r"$y$ [$\rm\AA$]")
    ax[2].set_title(r"(c) $U_{\rm aniso}/U_{\rm iso}$")
    fig.colorbar(im, ax=ax[2], shrink=0.85)

    fig.suptitle("Lu-dislocation interaction in Ta from the elastic dipole "
                 "tensor: anisotropic vs isotropic elasticity", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("stroh_field.png", dpi=140); fig.savefig("stroh_field.pdf", dpi=140)
    print("figure written: stroh_field.png")


if __name__ == "__main__":
    main()
