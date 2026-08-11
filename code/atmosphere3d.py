"""
3D Cottrell atmosphere WITH solute-solute interaction.

THE INCONSISTENCY THIS FIXES
    cottrell_kawasaki.py (2D) used the full lattice-gas Hamiltonian
        H = sum_i U_i c_i + J sum_<ij> c_i c_j
    and found the MC atmosphere systematically denser than the mean-field
    Fermi-Dirac profile.  depin3d.py (3D) then threw that away and sampled
    site occupations as independent Bernoulli draws from the SAME mean-field
    profile, i.e. J = 0.  Every 3D tau_c reported so far is therefore a lower
    bound.

WHY SEMI-GRAND-CANONICAL AND NOT KAWASAKI HERE
    Kawasaki is the right choice when the DYNAMICS matter (it conserves
    solute, as vacancy-mediated diffusion does).  Here we only need the
    EQUILIBRIUM distribution of a region that exchanges solute with a large
    bulk reservoir, which is a grand-canonical condition.  Single-site
    semi-grand-canonical flips sample exactly that distribution and equilibrate
    orders of magnitude faster than local exchange, which is diffusion-limited.
    A Kawasaki cross-check on a smaller box is included.

    mu is calibrated numerically in a field-free bulk box so that <c> = c0,
    rather than from the mean-field expression, so the far-field boundary
    condition is exact at every J.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

import depin3d as D
from depin3d import P, GPA, KB

Z_NN = 6          # simple-cubic proxy lattice (bcc has 8; see caveats)


# ------------------------------------------------------------------- fields
def u_field_2d(p=P):
    """z-independent interaction field, smooth core (see depin3d.u_field)."""
    ix = (np.arange(p.Nx) - p.Nx//2)*p.a
    iy = (np.arange(p.Ny) - p.Ny//2)*p.a
    X, Y = np.meshgrid(ix, iy, indexing="ij")
    return np.ascontiguousarray(D.u_field(X, Y, p)), X, Y


# --------------------------------------------------------------- SGCMC core
@njit(cache=True, fastmath=True)
def sgc_sweeps(c, U2d, J, mu, beta, n_sweeps):
    """
    Semi-grand-canonical Metropolis on H = sum U_i c_i + J sum_<ij> c_i c_j.
    c is int8 of shape (Nx, Ny, Nz); z is periodic, x and y are open
    (sites outside are treated as unoccupied, which is correct because the
    field has decayed there).
    """
    Nx, Ny, Nz = c.shape
    n_acc = 0
    for _ in range(n_sweeps):
        for _ in range(Nx*Ny*Nz):
            i = np.random.randint(0, Nx)
            j = np.random.randint(0, Ny)
            k = np.random.randint(0, Nz)
            ci = c[i, j, k]
            dn = 1 - 2*ci                       # +1 fill, -1 empty

            s = 0
            if i > 0:      s += c[i-1, j, k]
            if i < Nx-1:   s += c[i+1, j, k]
            if j > 0:      s += c[i, j-1, k]
            if j < Ny-1:   s += c[i, j+1, k]
            s += c[i, j, (k-1) % Nz]
            s += c[i, j, (k+1) % Nz]

            dH = (U2d[i, j] - mu)*dn + J*dn*s
            if dH <= 0.0 or np.random.random() < np.exp(-beta*dH):
                c[i, j, k] = 1 - ci
                n_acc += 1
    return n_acc/(n_sweeps*Nx*Ny*Nz)


@njit(cache=True, fastmath=True)
def kawasaki_sweeps_3d(c, U2d, J, beta, n_sweeps):
    """Conserved nearest-neighbour exchange, for the ensemble cross-check."""
    Nx, Ny, Nz = c.shape
    n_acc = 0
    n_att = 0
    for _ in range(n_sweeps):
        for _ in range(Nx*Ny*Nz):
            i = np.random.randint(0, Nx); j = np.random.randint(0, Ny)
            k = np.random.randint(0, Nz)
            d = np.random.randint(0, 6)
            i2, j2, k2 = i, j, k
            if   d == 0: i2 = i+1
            elif d == 1: i2 = i-1
            elif d == 2: j2 = j+1
            elif d == 3: j2 = j-1
            elif d == 4: k2 = (k+1) % Nz
            else:        k2 = (k-1) % Nz
            if i2 < 0 or i2 >= Nx or j2 < 0 or j2 >= Ny:
                continue
            ci = c[i, j, k]; cj = c[i2, j2, k2]
            if ci == cj:
                continue
            n_att += 1

            si = 0
            if i > 0:    si += c[i-1, j, k]
            if i < Nx-1: si += c[i+1, j, k]
            if j > 0:    si += c[i, j-1, k]
            if j < Ny-1: si += c[i, j+1, k]
            si += c[i, j, (k-1) % Nz] + c[i, j, (k+1) % Nz]
            si -= cj
            sj = 0
            if i2 > 0:    sj += c[i2-1, j2, k2]
            if i2 < Nx-1: sj += c[i2+1, j2, k2]
            if j2 > 0:    sj += c[i2, j2-1, k2]
            if j2 < Ny-1: sj += c[i2, j2+1, k2]
            sj += c[i2, j2, (k2-1) % Nz] + c[i2, j2, (k2+1) % Nz]
            sj -= ci

            dH = (U2d[i2, j2] - U2d[i, j])*(ci - cj) + J*(cj - ci)*(si - sj)
            if dH <= 0.0 or np.random.random() < np.exp(-beta*dH):
                c[i, j, k] = cj; c[i2, j2, k2] = ci
                n_acc += 1
    return n_acc/max(n_att, 1)


# ------------------------------------------------------------ mu calibration
def calibrate_mu(J, beta, c0, L=24, n_eq=120, n_meas=60, seed=1, tol=2e-4):
    """Bisect mu in a field-free box so that <c> = c0 exactly."""
    U0 = np.zeros((L, L), dtype=np.float64)
    kT = 1.0/beta
    mu_mf = kT*np.log(c0/(1-c0)) + Z_NN*J*c0
    lo, hi = mu_mf - 0.25, mu_mf + 0.25

    def conc(mu):
        np.random.seed(seed)
        c = (np.random.random((L, L, L)) < c0).astype(np.int8)
        c = np.ascontiguousarray(c)
        sgc_sweeps(c, U0, J, mu, beta, n_eq)
        acc = 0.0
        for _ in range(n_meas):
            sgc_sweeps(c, U0, J, mu, beta, 1)
            acc += c.mean()
        return acc/n_meas

    for _ in range(24):
        mid = 0.5*(lo + hi)
        if conc(mid) < c0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5*(lo + hi), mu_mf


# ---------------------------------------------------------------- pipeline
def equilibrate(J, p=P, T_age=None, n_eq=300, seed=0, verbose=True,
                theta=1.0, n_meas=0):
    """
    theta scales the field: theta=1 is the fully aged atmosphere, theta<1 a
    partially formed one.  (With J != 0 the old definition -- interpolating
    the occupation PROBABILITY -- is not well defined, because occupations are
    correlated.  Scaling the driving field is the natural generalisation and
    agrees with the old definition in the dilute mean-field limit.)

    If n_meas > 0, also returns the time-averaged occupancy <c>_t, which beats
    down the binomial noise of a single configuration.
    """
    T_age = T_age or p.T_age
    beta = 1.0/(KB*T_age)
    U2d, X, Y = u_field_2d(p)
    U2d = np.ascontiguousarray(theta*U2d)
    mu, mu_mf = calibrate_mu(J, beta, p.c0, seed=seed+1)

    np.random.seed(seed)
    c = np.ascontiguousarray(
        (np.random.random((p.Nx, p.Ny, p.Nz)) < p.c0).astype(np.int8))
    acc = sgc_sweeps(c, U2d, J, mu, beta, n_eq)

    cbar = None
    if n_meas:
        cbar = np.zeros((p.Nx, p.Ny))
        for _ in range(n_meas):
            sgc_sweeps(c, U2d, J, mu, beta, 2)
            cbar += c.mean(axis=2)
        cbar /= n_meas

    if verbose:
        print(f"    J = {J:+.3f} eV: mu = {mu:.4f} (mean field {mu_mf:.4f}), "
              f"acc = {acc:.3f}, <c> = {c.mean():.4f}, "
              f"peak c = {c.mean(axis=2).max():.3f}")
    return (c, U2d, X, Y, mu) if cbar is None else (c, U2d, X, Y, mu, cbar)


def solutes_from(c, X, Y, p=P):
    keep2d = (X**2 + Y**2) <= p.R_keep**2
    ii, jj, kk = np.nonzero(c & keep2d[:, :, None])
    return X[ii, jj], Y[ii, jj], kk.astype(np.int64)


def tau_c_from(c, X, Y, p=P):
    xs, ys, zs = solutes_from(c, X, Y, p)
    xgrid, V, Fpin = D.pinning_landscape(xs, ys, zs, p)
    Vbar = V.mean(axis=0)
    x0 = float(xgrid[np.argmin(Vbar)])
    tc, _, _ = D.tau_c_quasistatic(Fpin, xgrid, x0, p, tau_max_gpa=20.0)
    return tc, Vbar.min()/p.a, xs.size


# -------------------------------------------------------------------- main
def main(p=P):
    beta = 1.0/(KB*p.T_age)
    # 3D Ising: kTc = 4.511 * J_ising, J_ising = |omega|/4
    print(f"T_age = {p.T_age:.0f} K, kT = {1000/beta:.1f} meV, c0 = {p.c0}")
    for om in [0.02, 0.04, 0.06]:
        Tc = 4.511*(om/4)/KB
        print(f"  omega = {om:.2f} eV -> bulk demixing Tc = {Tc:.0f} K "
              f"({'above' if p.T_age > Tc else 'BELOW'} T_age)")
    print()

    omegas = [0.0, 0.02, 0.04, 0.06]
    n_seed = 3
    res = {}

    print(f"{'omega':>6} {'J':>7} {'N_sol':>7} {'depth':>8} {'tau_c [GPa]':>16}")
    print("-" * 52)
    for om in omegas:
        J = -om                       # J_eff = -omega (see depin3d docstring)
        tcs, dep, nsl, csave = [], 0.0, 0, None
        for s in range(n_seed):
            c, U2d, X, Y, mu = equilibrate(J, p, seed=100*s + 7, verbose=(s == 0))
            tc, dep, nsl = tau_c_from(c, X, Y, p)
            tcs.append(tc)
            if s == 0:
                csave = c.mean(axis=2)
        tcs = np.array(tcs)
        res[om] = dict(tau=float(tcs.mean()), sd=float(tcs.std(ddof=1)),
                       depth=dep, nsol=nsl, cmap=csave)
        print(f"{om:6.2f} {J:+7.3f} {nsl:7d} {dep:8.3f} "
              f"{tcs.mean():8.3f} +/-{tcs.std(ddof=1):5.3f}")

    base = res[0.0]["tau"]
    print(f"\nenhancement over the J = 0 (Bernoulli) atmosphere used so far:")
    for om in omegas:
        print(f"  omega = {om:.2f} eV : {res[om]['tau']/base:5.2f}x")

    validate(p, res)
    return res


# ------------------------------------------------------------- validation
def validate(p=P, res=None):
    print("\n" + "="*52)
    print("VALIDATION")
    beta = 1.0/(KB*p.T_age)

    # 1. J = 0 SGCMC must reproduce the analytic Fermi-Dirac profile.
    #    Compare against the BINOMIAL noise floor: <c>_z averages Nz slices
    #    and n_meas samples, so a per-site sd of sqrt(c(1-c)/(Nz*n_meas)) is
    #    expected even for a perfect match.  Comparing raw max deviations
    #    (as a first version of this test did) just measures that noise.
    n_meas = 60
    c, U2d, X, Y, mu, cm = equilibrate(0.0, p, seed=11, verbose=False,
                                       n_meas=n_meas)
    e = np.exp(-U2d/(KB*p.T_age))
    fd = p.c0*e/(1 - p.c0 + p.c0*e)
    m = (X**2 + Y**2 < 40.0**2) & (fd > 0.02)
    dev = cm[m] - fd[m]
    noise = np.sqrt(fd[m]*(1-fd[m])/(p.Nz*n_meas)).mean()
    zscore = np.abs(dev).mean()/noise
    print(f"  J=0 SGCMC vs analytic Fermi-Dirac:")
    print(f"    mean |dc| = {np.abs(dev).mean():.4f}, "
          f"binomial noise floor = {noise:.4f}  -> {zscore:.1f} sigma")
    print(f"    bias <dc> = {dev.mean():+.4f}   "
          f"({'PASS' if abs(dev.mean()) < 3*noise else 'FAIL'})")

    # 2. SGCMC and Kawasaki must agree on the equilibrium structure
    q = type("Q", (P,), dict(Nx=32, Ny=32, Nz=32))
    J = -0.04
    cs, U2s, Xs, Ys, mus = equilibrate(J, q, n_eq=400, seed=5, verbose=False)
    np.random.seed(5)
    nsol = int(cs.sum())
    flat = np.zeros(q.Nx*q.Ny*q.Nz, dtype=np.int8); flat[:nsol] = 1
    np.random.shuffle(flat)
    ck = np.ascontiguousarray(flat.reshape(q.Nx, q.Ny, q.Nz))
    kawasaki_sweeps_3d(ck, U2s, J, beta, 4000)
    a = cs.mean(axis=2); b = ck.mean(axis=2)
    mm = (Xs**2 + Ys**2 < 25.0**2)
    print(f"  SGCMC vs Kawasaki (32^3, omega=0.04): "
          f"peak c {a.max():.3f} vs {b.max():.3f}, "
          f"mean |dc| in core region = {np.abs(a[mm]-b[mm]).mean():.4f}")

    if res:
        make_figure(res, p, X, Y, cm, fd)


def make_figure(res, p, X, Y, cm, fd):
    oms = sorted(res)
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.4))
    ext = [X.min(), X.max(), Y.min(), Y.max()]

    for i, om in enumerate([0.0, 0.06]):
        im = ax[i].imshow(res[om]["cmap"].T, origin="lower", cmap="magma",
                          vmin=0, vmax=1, extent=ext)
        ax[i].set_xlim(-35, 35); ax[i].set_ylim(-35, 35)
        ax[i].set_xlabel(r"$x$ [$\rm\AA$]"); ax[i].set_ylabel(r"$y$ [$\rm\AA$]")
        ax[i].set_title(rf"({chr(97+i)}) $\langle c\rangle_z$, "
                        rf"$\omega$ = {om:.2f} eV")
        fig.colorbar(im, ax=ax[i], shrink=0.85)

    mid = p.Nx//2
    yv = (np.arange(p.Ny) - p.Ny//2)*p.a
    sel = yv < -3.0
    ax[2].plot(-yv[sel], fd[mid, sel], "k-", lw=2.5, alpha=0.4,
               label="analytic Fermi-Dirac")
    for om in oms:
        ax[2].plot(-yv[sel], res[om]["cmap"][mid, sel], "o-", ms=3, lw=1.4,
                   label=rf"$\omega$={om:.2f}")
    ax[2].set_xscale("log"); ax[2].set_xlim(3, 90)
    ax[2].set_xlabel(r"$|y|$ on the tension axis [$\rm\AA$]")
    ax[2].set_ylabel(r"$c_{\rm Lu}$")
    ax[2].set_title("(c) radial profile")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    t = np.array([res[o]["tau"] for o in oms])
    e = np.array([res[o]["sd"] for o in oms])
    ax[3].errorbar(oms, t, yerr=e, fmt="o-", lw=2, capsize=4)
    ax[3].axhline(t[0], color="0.5", ls="--", lw=1.2,
                  label=r"$J=0$ (Bernoulli), used so far")
    ax[3].axhline(p.mu/30/GPA, color="0.35", ls="-.", lw=1.3,
                  label=r"$\mu/30$")
    ax[3].set_xlabel(r"$\omega = 2\epsilon_{AB}-\epsilon_{AA}-\epsilon_{BB}$ [eV]")
    ax[3].set_ylabel(r"$\tau_c$ (0 K) [GPa]")
    ax[3].set_title("(d) effect of solute-solute interaction")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)

    fig.suptitle("3D Cottrell atmosphere with solute-solute interaction "
                 "(semi-grand-canonical MC)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("atmosphere3d.png", dpi=140)
    print("\nfigure written: atmosphere3d.png")


if __name__ == "__main__":
    main()
