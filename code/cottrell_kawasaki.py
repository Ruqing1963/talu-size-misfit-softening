"""
Cottrell atmosphere formation at an edge dislocation in a supersaturated Ta(Lu)
solid solution.

Model
-----
2D lattice gas on a square lattice (projection normal to the dislocation line).
    c_i = 1  -> Lu (oversized solute, dV = +11.45 A^3)
    c_i = 0  -> Ta (matrix)

    H = sum_i U_disl(r_i) c_i  +  J sum_<ij> c_i c_j
    U_disl(r,theta) = A sin(theta) / r ,  A = (1+nu) mu b dV / [3 pi (1-nu)]
    J = -omega,  omega = 2 eps_AB - eps_AA - eps_BB  (omega > 0 -> clustering)

Dynamics
--------
Conserved (canonical) Kawasaki spin-exchange with Metropolis acceptance.
Only nearest-neighbour exchanges are attempted, so the trajectory is a genuine
vacancy-free diffusion proxy, NOT Glauber / non-conserved dynamics.

Units: eV and Angstrom throughout.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
from numba import njit

KB = 8.617333262e-5  # eV / K


# ----------------------------------------------------------------- parameters
class P:
    # lattice
    L        = 192          # sites per side
    a        = 2.863        # A  (= |b| for bcc Ta, a0*sqrt(3)/2)

    # elastic coupling
    A_el     = 3.042        # eV*A   -> U = A_el * sin(theta)/r
    r_core   = 2.0          # in units of a; linear elasticity invalid inside
    U_cap    = 1.20         # eV; hard cap standing in for DFT core energy
    R_cut_f  = 0.45         # taper U to 0 beyond R_cut_f * L * a (PBC safety)

    # thermodynamics
    c0       = 0.02         # nominal Lu atomic fraction
    T        = 600.0        # K
    omega    = 0.06         # eV/bond, >0 => Lu-Ta unlike bonds unfavourable
    z        = 4            # coordination of the 2D square lattice

    # Monte Carlo
    n_equil  = 3000         # sweeps (1 sweep = L*L exchange attempts)
    n_sample = 2000
    n_meas   = 40           # measure every n_meas sweeps
    seed     = 20260807


# ------------------------------------------------------------- stress field
def build_field(p):
    """U_disl on the lattice, with core regularisation and far-field taper."""
    idx = (np.arange(p.L) - p.L // 2) * p.a
    X, Y = np.meshgrid(idx, idx, indexing="ij")
    R = np.hypot(X, Y)

    U = np.zeros_like(R)
    live = R > p.r_core * p.a
    U[live] = p.A_el * Y[live] / R[live] ** 2      # A sin(theta)/r  ==  A y/r^2
    U = np.clip(U, -p.U_cap, p.U_cap)

    U[R > p.R_cut_f * p.L * p.a] = 0.0             # avoid PBC image artefacts
    return U, X, Y, R


# --------------------------------------------------------------- MC kernel
@njit(cache=True, fastmath=True)
def _nnsum(c, i, j, L):
    return (c[(i + 1) % L, j] + c[(i - 1) % L, j]
            + c[i, (j + 1) % L] + c[i, (j - 1) % L])


@njit(cache=True, fastmath=True)
def kawasaki_sweeps(c, U, J, beta, n_sweeps):
    """n_sweeps sweeps of conserved Kawasaki + Metropolis. Returns accept rate."""
    L = c.shape[0]
    n_att = 0
    n_acc = 0

    for _ in range(n_sweeps):
        for _ in range(L * L):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            ci = c[i, j]

            d = np.random.randint(0, 4)
            if d == 0:
                i2, j2 = (i + 1) % L, j
            elif d == 1:
                i2, j2 = (i - 1) % L, j
            elif d == 2:
                i2, j2 = i, (j + 1) % L
            else:
                i2, j2 = i, (j - 1) % L

            cj = c[i2, j2]
            if ci == cj:            # identical species: swap is the identity
                continue
            n_att += 1

            # neighbour sums EXCLUDING the exchange partner
            Si = _nnsum(c, i, j, L) - cj
            Sj = _nnsum(c, i2, j2, L) - ci

            dE = (U[i2, j2] - U[i, j]) * (ci - cj) + J * (cj - ci) * (Si - Sj)

            if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
                c[i, j] = cj
                c[i2, j2] = ci
                n_acc += 1

    return n_acc / max(n_att, 1)


# ------------------------------------------------- Warren-Cowley alpha_1 map
@njit(cache=True)
def unlike_bonds(c):
    """u[i] = number of Lu-Ta nearest-neighbour bonds seen from site i (0 if Ta)."""
    L = c.shape[0]
    u = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            if c[i, j] == 1:
                u[i, j] = 4.0 - _nnsum(c, i, j, L)
    return u


def alpha1_window(cacc, uacc, w, n_samples, z=4, n_min=8.0):
    """
    Time-averaged, sliding-window alpha_1 = 1 - P(Ta|Lu)/c_Ta.

    cacc, uacc : per-site time averages of c_i and of the unlike-bond count u_i.
    w          : window edge in lattice sites (odd).
    n_samples  : number of independent configurations averaged (for statistics).
    n_min      : minimum effective atom count of EITHER species to report a value.

    Windows containing too few atoms of one species carry no usable pair
    statistics and are returned as NaN rather than as spurious extreme values.
    """
    from scipy.ndimage import uniform_filter

    nsite = w * w
    cw = uniform_filter(cacc, size=w, mode="wrap")          # local c_Lu
    uw = uniform_filter(uacc, size=w, mode="wrap")          # <u> per site

    nLu = cw * nsite
    nLuTa = uw * nsite
    cTa = 1.0 - cw

    eff_Lu = nLu * n_samples                                # effective counts
    eff_Ta = cTa * nsite * n_samples

    out = np.full_like(cw, np.nan)
    ok = (eff_Lu > n_min) & (eff_Ta > n_min)
    out[ok] = 1.0 - (nLuTa[ok] / (z * nLu[ok])) / cTa[ok]
    return out, cw


# ------------------------------------------------------- analytic references
def fd_profile(U, c0, T):
    """McLean-Langmuir / Fermi-Dirac equilibrium occupancy."""
    e = np.exp(-U / (KB * T))
    return c0 * e / (1.0 - c0 + c0 * e)


def alpha1_qca(cA, omega, T):
    """Quasi-chemical alpha_1 at local composition cA."""
    cA = np.clip(cA, 1e-9, 1 - 1e-9)
    cB = 1.0 - cA
    lam = np.exp(-omega / (KB * T))
    if abs(1.0 - lam) < 1e-12:
        x = cA * cB
    else:
        x = (-lam + np.sqrt(lam**2 + 4 * lam * (1 - lam) * cA * cB)) / (2 * (1 - lam))
    return 1.0 - x / (cA * cB)


# ------------------------------------------------------------------- driver
def run(p=P):
    np.random.seed(p.seed)
    U, X, Y, R = build_field(p)
    beta = 1.0 / (KB * p.T)
    J = -p.omega

    # random initial solid solution at the nominal concentration
    N = p.L * p.L
    nLu = int(round(p.c0 * N))
    c = np.zeros(N, dtype=np.int8)
    c[:nLu] = 1
    np.random.shuffle(c)
    c = np.ascontiguousarray(c.reshape(p.L, p.L))

    # ---- equilibration
    acc = kawasaki_sweeps(c, U, J, beta, p.n_equil)
    print(f"equilibration acceptance rate = {acc:.3f}")

    # ---- production: accumulate <c(r)> and unlike-bond statistics
    cbar = np.zeros((p.L, p.L))
    ubar = np.zeros((p.L, p.L))
    nblk = p.n_sample // p.n_meas
    for _ in range(nblk):
        kawasaki_sweeps(c, U, J, beta, p.n_meas)
        cbar += c
        ubar += unlike_bonds(c)
    cbar /= nblk
    ubar /= nblk

    # ---- diagnostics
    Uc = KB * p.T * np.log((1 - p.c0) / p.c0)
    Rc = p.A_el / Uc
    print(f"T = {p.T:.0f} K   kT = {KB*p.T*1e3:.2f} meV")
    print(f"condensation-circle diameter R_c = {Rc:.1f} A = {Rc/p.a:.2f} b")
    print(f"peak <c_Lu> = {cbar.max():.3f}  (nominal c0 = {p.c0})")

    return c, cbar, ubar, U, X, Y, Rc


# --------------------------------------------------------------------- plots
def make_figure(c, cbar, ubar, U, X, Y, Rc, p=P, fname="cottrell_Ta_Lu.png"):
    ext = [X.min(), X.max(), Y.min(), Y.max()]
    win = 40.0  # A, zoom half-width

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 11.8))

    # tangent circle of diameter Rc, centre (0, -Rc/2)
    th = np.linspace(0, 2 * np.pi, 400)
    cx = 0.5 * Rc * np.cos(th)
    cy = -0.5 * Rc + 0.5 * Rc * np.sin(th)

    # (a) elastic field
    im = ax[0, 0].imshow(U.T, origin="lower", extent=ext, cmap="RdBu_r",
                         vmin=-p.U_cap, vmax=p.U_cap)
    ax[0, 0].plot(cx, cy, "k--", lw=1.5, label=r"$c=1/2$ isoline")
    ax[0, 0].set_title(r"(a) $U_{\rm disl}=A\sin\theta/r$   [eV]")
    ax[0, 0].legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax[0, 0], shrink=0.82)

    # (b) time-averaged concentration
    im = ax[0, 1].imshow(cbar.T, origin="lower", extent=ext, cmap="magma",
                         vmin=0, vmax=1)
    ax[0, 1].plot(cx, cy, "c--", lw=1.5)
    ax[0, 1].set_title(rf"(b) MC $\langle c_{{\rm Lu}}\rangle$, $T$={p.T:.0f} K, $c_0$={p.c0}")
    fig.colorbar(im, ax=ax[0, 1], shrink=0.82)

    for a in ax[0]:
        a.set_xlim(-win, win); a.set_ylim(-win, win)
        a.set_xlabel(r"$x$ [$\rm\AA$]"); a.set_ylabel(r"$y$ [$\rm\AA$]")
        a.axhline(0, color="0.5", lw=0.6)

    # sliding-window Warren-Cowley map (used by panels c and d)
    w = 7
    nsmp = p.n_sample // p.n_meas
    a1, cw = alpha1_window(cbar, ubar, w, nsmp, p.z)

    # (c) profile along the tension axis (x=0, y<0) vs Fermi-Dirac
    mid = p.L // 2
    yv = (np.arange(p.L) - mid) * p.a
    sel = yv < -p.r_core * p.a
    cFD = fd_profile(U, p.c0, p.T)
    ax[1, 0].plot(-yv[sel], cbar[mid, sel], "o", ms=4, label="Kawasaki MC")
    ax[1, 0].plot(-yv[sel], cFD[mid, sel], "-", lw=2, label="Fermi-Dirac (mean field)")
    ax[1, 0].axvline(Rc, color="k", ls=":", label=rf"$R_c$ = {Rc:.0f} $\rm\AA$")
    ax[1, 0].axhline(p.c0, color="0.6", ls="--", lw=1)
    ax[1, 0].set_xscale("log"); ax[1, 0].set_xlim(p.r_core * p.a, 90)
    ax[1, 0].set_xlabel(r"$|y|$ along tension axis [$\rm\AA$]")
    ax[1, 0].set_ylabel(r"$c_{\rm Lu}$")
    ax[1, 0].set_title(r"(c) radial profile, $\theta=-\pi/2$")
    ax[1, 0].legend(fontsize=9, loc="center right"); ax[1, 0].grid(alpha=0.3)

    axr = ax[1, 0].twinx()
    axr.plot(-yv[sel], a1[mid, sel], "s-", ms=3, lw=1.2, color="tab:green",
             alpha=0.8, label=r"$\alpha_1$")
    axr.set_ylabel(r"$\alpha_1$", color="tab:green")
    axr.tick_params(axis="y", labelcolor="tab:green")
    axr.axhline(0, color="tab:green", ls=":", lw=0.8)

    # (d) Warren-Cowley alpha_1, sliding window
    vlim = 0.5   # fixed symmetric scale: alpha_1 in [-1, 1] by definition
    im = ax[1, 1].imshow(a1.T, origin="lower", extent=ext, cmap="PuOr",
                         vmin=-vlim, vmax=vlim)
    ax[1, 1].plot(cx, cy, "k--", lw=1.5)
    ax[1, 1].contour(X, Y, cw, levels=[0.5], colors="g", linewidths=1.2)
    ax[1, 1].set_xlim(-win, win); ax[1, 1].set_ylim(-win, win)
    ax[1, 1].set_xlabel(r"$x$ [$\rm\AA$]"); ax[1, 1].set_ylabel(r"$y$ [$\rm\AA$]")
    ax[1, 1].set_title(rf"(d) Warren-Cowley $\alpha_1$ ({w}$\times${w} window)"
                       "\n" r"green: MC $c_{\rm Lu}=0.5$ contour")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.82)

    fig.suptitle("Cottrell atmosphere in supersaturated Ta(Lu):  "
                 r"conserved Kawasaki MC vs. mean-field theory", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=150)
    fig.savefig(fname.replace('.png', '.pdf'))
    print(f"figure written: {fname}")
    return a1, cw


if __name__ == "__main__":
    c, cbar, ubar, U, X, Y, Rc = run()
    a1, cw = make_figure(c, cbar, ubar, U, X, Y, Rc)

    # alpha_1 vs local composition: MC vs quasi-chemical prediction
    ok = np.isfinite(a1) & (cw > 0.01)
    cc = np.linspace(0.01, 0.99, 120)
    plt.figure(figsize=(6.6, 4.8))
    plt.plot(cw[ok].ravel(), a1[ok].ravel(), ".", ms=3, alpha=0.25,
             label="MC sliding windows")
    plt.plot(cc, [alpha1_qca(v, P.omega, P.T) for v in cc], "r-", lw=2,
             label="quasi-chemical (Guggenheim)")
    plt.axhline(0, color="0.5", lw=0.8)
    plt.xlabel(r"local $c_{\rm Lu}$"); plt.ylabel(r"$\alpha_1$")
    plt.title(r"$\alpha_1$ peaks at intermediate $c$, $\to 0$ in the saturated core")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("alpha1_vs_c.png", dpi=150); plt.savefig("alpha1_vs_c.pdf")
    print("figure written: alpha1_vs_c.png")
