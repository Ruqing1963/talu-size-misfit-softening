"""
How much does tau_c depend on the two things we have not yet computed from DFT?

  (1) U_cap  -- the core binding energy, currently a placeholder
  (2) the elastic model -- isotropic vs anisotropic (Stroh)

Both are swept here by feeding depin3d's machinery a TABULATED interaction
field U(dx, dy) built from the elastic dipole tensor, instead of the hard-coded
isotropic  A sin(theta)/r.

The answer to (1) sets how much DFT accuracy is actually needed; the answer to
(2) says whether anisotropic elasticity is worth carrying.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

import depin3d as D
from depin3d import P, GPA, KB, gamma_edge
import stroh_field as SF


# --------------------------------------------------------- tabulated fields
class FieldTable:
    """U(dx, dy) on a regular grid, ready for fast bilinear lookup in numba."""

    def __init__(self, U, dx_lo, dy_lo, h):
        self.U = np.ascontiguousarray(U)
        self.dx_lo, self.dy_lo, self.h = dx_lo, dy_lo, h


def build_table(kind, U_cap, p=P, h=0.20, dx_max=210.0, dy_max=70.0):
    """
    kind = 'aniso' : Stroh with the real Ta cubic constants
    kind = 'iso'   : Stroh with the Voigt-averaged isotropic constants
                     (identical to the closed form A sin(theta)/r)
    """
    C11, C12, C44 = 266.0, 158.2, 87.4
    K = (C11 + 2*C12)/3.0
    Omega_rel = 11.45
    P_scalar = K*GPA*Omega_rel
    Qrot = SF.dislocation_frame()
    bvec = np.array([p.b, 0.0, 0.0])

    if kind == "aniso":
        C = SF.cubic_C(C11*GPA, C12*GPA, C44*GPA)
    else:
        mu_V = (C11 - C12 + 3*C44)/5.0
        nu_V = (3*K - 2*mu_V)/(2*(3*K + mu_V))
        C = SF.isotropic_C(mu_V*GPA, nu_V)
    st = SF.Stroh(SF.rotate_C(C, Qrot))

    gx = np.arange(-dx_max, dx_max + 1e-9, h)
    gy = np.arange(-dy_max, dy_max + 1e-9, h)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    U = SF.interaction_field(X, Y, P_scalar, st, bvec, U_cap)
    return FieldTable(U, gx[0], gy[0], h)


@njit(cache=True, fastmath=True)
def _lookup(U, dx_lo, dy_lo, h, dx, dy):
    gi = (dx - dx_lo)/h
    gj = (dy - dy_lo)/h
    ni, nj = U.shape
    if gi < 0.0 or gi >= ni - 1 or gj < 0.0 or gj >= nj - 1:
        return 0.0
    i = int(gi); j = int(gj)
    wi = gi - i; wj = gj - j
    return (U[i, j]*(1-wi)*(1-wj) + U[i+1, j]*wi*(1-wj)
            + U[i, j+1]*(1-wi)*wj + U[i+1, j+1]*wi*wj)


@njit(cache=True, fastmath=True)
def _accumulate_V_tab(xs, ys, zs, xgrid, Nz, U, dx_lo, dy_lo, h):
    """Same as depin3d._accumulate_V but reading a tabulated U(dx, dy)."""
    V = np.zeros((Nz, xgrid.size))
    for s in range(xs.size):
        k = zs[s]
        xi = xs[s]; yi = ys[s]
        for m in range(xgrid.size):
            V[k, m] += _lookup(U, dx_lo, dy_lo, h, xi - xgrid[m], yi)
    return V


# ------------------------------------------------------------- one full case
def run_case(kind, U_cap, theta, seed, p=P):
    tab = build_table(kind, U_cap, p)

    # --- occupancy from the SAME field (so ageing is consistent)
    ix = (np.arange(p.Nx) - p.Nx//2)*p.a
    iy = (np.arange(p.Ny) - p.Ny//2)*p.a
    X, Y = np.meshgrid(ix, iy, indexing="ij")
    U2d = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U2d[i, j] = _lookup(tab.U, tab.dx_lo, tab.dy_lo, tab.h,
                                X[i, j], Y[i, j])
    e = np.exp(-U2d/(KB*p.T_age))
    c_eq = p.c0*e/(1.0 - p.c0 + p.c0*e)
    occ = p.c0 + theta*(c_eq - p.c0)

    rng = np.random.default_rng(seed)
    keep = (X**2 + Y**2) <= p.R_keep**2
    xs, ys, zs = [], [], []
    for k in range(p.Nz):
        draw = (rng.random(occ.shape) < occ) & keep
        gx, gy = np.nonzero(draw)
        xs.append(X[gx, gy]); ys.append(Y[gx, gy]); zs.append(np.full(gx.size, k))
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    zs = np.concatenate(zs).astype(np.int64)

    xgrid = np.arange(p.x_lo, p.x_hi + 1e-9, p.dx_grid)
    V = _accumulate_V_tab(xs, ys, zs, xgrid, p.Nz,
                          tab.U, tab.dx_lo, tab.dy_lo, tab.h)
    Fpin = np.ascontiguousarray(-np.gradient(V, p.dx_grid, axis=1))
    Vbar = V.mean(axis=0)
    x0 = float(xgrid[np.argmin(Vbar)])

    tc, _, _ = D.tau_c_quasistatic(Fpin, xgrid, x0, p)
    return tc, Vbar.min()/p.a, xs.size


# -------------------------------------------------------------------- driver
def main(p=P):
    tau_theo = p.mu/30.0/GPA
    caps = [0.30, 0.50, 0.70, 1.00, 1.50]
    thetas = [0.0, 0.25]
    n_seed = 4

    print("tau_c [GPa], mean over "
          f"{n_seed} solute realisations;  mu/30 = {tau_theo:.2f} GPa\n")
    print(f"{'U_cap':>6} {'field':>6} {'theta':>6} {'depth':>8} "
          f"{'N_sol':>7} {'tau_c':>16}")
    print("-" * 56)

    res = {}
    for kind in ["aniso", "iso"]:
        for th in thetas:
            for cap in caps:
                tcs, dep, nsl = [], 0.0, 0
                for s in range(n_seed):
                    tc, dep, nsl = run_case(kind, cap, th,
                                            p.seed + 313*s + int(100*cap), p)
                    tcs.append(tc)
                tcs = np.array(tcs)
                res[(kind, th, cap)] = (tcs.mean(), tcs.std())
                print(f"{cap:6.2f} {kind:>6} {th:6.2f} {dep:8.3f} {nsl:7d} "
                      f"{tcs.mean():8.3f} +/-{tcs.std():5.3f}")
        print()

    # --- power-law fit tau_c ~ U_cap^n
    print("scaling of tau_c with the core binding energy:")
    for kind in ["aniso", "iso"]:
        for th in thetas:
            y = np.array([res[(kind, th, c)][0] for c in caps])
            n = np.polyfit(np.log(caps), np.log(y), 1)[0]
            print(f"  {kind:>6}, theta={th:.2f}:  tau_c ~ U_cap^{n:.2f}")

    # --- anisotropy penalty at the reference cap
    print("\neffect of using isotropic instead of anisotropic elasticity:")
    for th in thetas:
        for cap in [0.30, 0.70, 1.50]:
            a = res[("aniso", th, cap)][0]
            i = res[("iso", th, cap)][0]
            print(f"  theta={th:.2f}, U_cap={cap:.2f}:  "
                  f"iso/aniso = {i/a:.3f}  ({100*(i/a - 1):+.1f} %)")

    make_figure(caps, thetas, res, tau_theo)
    return res


def make_figure(caps, thetas, res, tau_theo):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.9))
    caps = np.array(caps)

    style = {("aniso", 0.0): ("o-", "tab:blue"), ("aniso", 0.25): ("o-", "tab:red"),
             ("iso", 0.0): ("s--", "tab:cyan"), ("iso", 0.25): ("s--", "tab:orange")}
    for (kind, th), (fmt, col) in style.items():
        y = np.array([res[(kind, th, c)][0] for c in caps])
        e = np.array([res[(kind, th, c)][1] for c in caps])
        ax[0].errorbar(caps, y, yerr=e, fmt=fmt, color=col, capsize=3, lw=1.8,
                       label=rf"{kind}, $\theta$={th}")
    ax[0].axhline(tau_theo, color="0.35", ls="-.", lw=1.4,
                  label=r"$\mu/30$")
    ax[0].axhspan(tau_theo, 1e3, color="0.9", zorder=0)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylim(0.2, 12)
    ax[0].set_xlabel(r"core cap $U_{\rm cap}$ [eV]")
    ax[0].set_ylabel(r"$\tau_c$ (0 K) [GPa]")
    ax[0].set_title("(a) sensitivity to the core binding energy")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    for th, col in [(0.0, "tab:blue"), (0.25, "tab:red")]:
        r = np.array([res[("iso", th, c)][0]/res[("aniso", th, c)][0]
                      for c in caps])
        ax[1].plot(caps, r, "o-", color=col, lw=1.8, label=rf"$\theta$={th}")
    ax[1].axhline(1.0, color="k", lw=1.0)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$U_{\rm cap}$ [eV]")
    ax[1].set_ylabel(r"$\tau_c^{\rm iso}\,/\,\tau_c^{\rm aniso}$")
    ax[1].set_title("(b) cost of assuming isotropic elasticity")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    fig.suptitle("What still has to come from DFT: core binding energy and "
                 "elastic anisotropy", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("sensitivity.png", dpi=140)
    print("\nfigure written: sensitivity.png")


if __name__ == "__main__":
    np.random.seed(P.seed)
    main()
