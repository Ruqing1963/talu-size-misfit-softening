"""
3D depinning of a flexible edge dislocation from a frozen Cottrell atmosphere
in supersaturated Ta(Lu).

WHY THE ATMOSPHERE IS FROZEN
    Q_diff(Lu in Ta) >~ 3.5 eV  =>  D(600 K) ~ 1e-34 m^2/s.  One atomic hop
    takes ~1e14 years.  Below ~0.4 T_m no coupled diffusion/glide KMC is
    needed: sample the solute field once, freeze it, then do line dynamics.

THREE-STAGE ARCHITECTURE
    STAGE A  sample the frozen 3D solute field.  The ageing fraction theta
             interpolates between a random solid solution (theta=0, the
             Varvenne-Curtin limit) and the fully equilibrated Fermi-Dirac
             atmosphere (theta=1).                                    [once]
    STAGE B  tabulate the pinning landscape V(x,z) and F = -dV/dx on a fine
             x-grid.  This removes the O(N_solute) sum from the inner loop.
                                                                      [once]
    STAGE C  flexible-line dynamics, two independent measurements:
             (C1) QUASI-STATIC ramp, zero noise, full relaxation at every
                  stress -> tau_c(0 K).  Drag-free: B only sets convergence
                  speed, not the answer.
             (C2) FIXED-STRESS Langevin first passage at finite T -> escape
                  time t(tau), testing the dE* ~ (1-tau/tau_c)^{3/2} scaling
                  derived from the string-nucleation saddle point.

Geometry: line along z, glide plane y = 0, Burgers vector along x.
Units:    eV, Angstrom.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

KB = 8.617333262e-5          # eV/K
GPA = 6.2415e-3              # 1 GPa in eV/A^3   (=> 1 eV/A^3 = 160.2 GPa)


# ----------------------------------------------------------------- parameters
class P:
    # --- crystal / elasticity (bcc Ta)
    a       = 2.863          # A, = |b|
    b       = 2.863
    mu      = 69.0 * GPA     # eV/A^3
    nu      = 0.34
    A_el    = 3.042          # eV*A,  U = A sin(theta)/r
    U_cap   = 0.70           # eV, peak binding; sets r_c = A/(2 U_cap)
    lnRr0   = 6.9            # ln(R/r0)

    # --- solute field
    c0      = 0.02
    Nx      = 64
    Ny      = 64
    Nz      = 128            # line length L = Nz*a = 366 A
    R_keep  = 60.0           # A
    T_age   = 900.0          # K, temperature the FD profile is evaluated at

    # --- pinning-landscape grid
    x_lo    = -60.0
    x_hi    = 140.0
    dx_grid = 0.20

    # --- dynamics
    Bdrag   = 1.0
    dt      = 1.00           # stability: dt < B * a / (2 Gamma) = 1.52
    tol_F   = 1e-3           # eV/A, force-balance tolerance
    n_relax = 12000          # max relaxation iterations per stress step
    d_runaway = 45.0         # A, advance within ONE stress step = depinned
    v_stall = 2e-5           # A/iteration, below this the line counts as pinned
    chunk   = 500            # iterations between stall checks
    n_tau   = 500            # stress-ramp resolution

    seed    = 20260807


def gamma_edge(p=P):
    """Dewit-Koehler line tension of an edge dislocation, eV/A.

    Gamma = E + d2E/dtheta2 = mu b^2 (1-2nu) ln(R/r0) / [4 pi (1-nu)]
    The factor (1-2nu) = 0.32 for Ta makes an edge dislocation about half as
    stiff as the naive mu b^2 / 2 estimate.
    """
    return p.mu * p.b**2 * (1.0 - 2.0*p.nu) * p.lnRr0 / (4.0*np.pi*(1.0 - p.nu))


# ================================================== STAGE A : solute field ===
def r_reg(p=P):
    """Core regularisation length, fixed by the peak binding energy U_cap."""
    return p.A_el/(2.0*p.U_cap)


def u_field(X, Y, p=P):
    """
    Smoothly core-regularised size interaction:

        U = A y / (x^2 + y^2 + r_c^2),      r_c = A / (2 U_cap)

    -> A sin(theta)/r at large r, -> 0 at the core, peak value exactly U_cap,
    and analytic everywhere.

    The earlier hard cutoff (U set to +/-U_cap inside r_core, then clipped)
    put a ~0.4 eV STEP into U(x).  Its numerical derivative is a spike of
    height ~1/dx and width ~dx, so the measured tau_c drifted by 57 % between
    dx_grid = 0.05 and 0.2 A and was not even monotonic.  See convergence.py.
    """
    rc2 = r_reg(p)**2
    return p.A_el * Y / (X*X + Y*Y + rc2)


def occupancy_profile(theta, p=P):
    """
    Site occupation probability for ageing fraction theta in [0, 1].

        theta = 0     uniform c0                    (random alloy, VC limit)
        theta = 1     full Fermi-Dirac atmosphere at T_age
        0 < theta < 1 linear interpolation: a proxy for a diffusion-limited
                      partially formed atmosphere.  It is a proxy, not a
                      solution of the Cottrell kinetics -- see caveats.
    """
    ix = (np.arange(p.Nx) - p.Nx//2) * p.a
    iy = (np.arange(p.Ny) - p.Ny//2) * p.a
    X, Y = np.meshgrid(ix, iy, indexing="ij")
    U2d = u_field(X, Y, p)

    e = np.exp(-U2d / (KB * p.T_age))
    c_eq = p.c0 * e / (1.0 - p.c0 + p.c0 * e)
    return p.c0 + theta * (c_eq - p.c0), U2d, X, Y


def build_atmosphere(theta, p=P, rng=None):
    """Draw one frozen 3D configuration.  Returns solute coords + slice index."""
    rng = rng or np.random.default_rng(p.seed)
    occ2d, U2d, X, Y = occupancy_profile(theta, p)
    keep = (X**2 + Y**2) <= p.R_keep**2

    xs, ys, zs = [], [], []
    for k in range(p.Nz):
        draw = (rng.random(occ2d.shape) < occ2d) & keep
        gx, gy = np.nonzero(draw)
        xs.append(X[gx, gy]); ys.append(Y[gx, gy])
        zs.append(np.full(gx.size, k))

    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(zs).astype(np.int64), occ2d, X, Y)


# =========================================== STAGE B : pinning landscape =====
@njit(cache=True, fastmath=True)
def _accumulate_V(xs, ys, zs, xgrid, Nz, A_el, rcore2, Ucap):
    """V[k, m] = sum over solutes in slice k of U(x_i - xgrid[m], y_i)."""
    V = np.zeros((Nz, xgrid.size))
    for s in range(xs.size):
        k = zs[s]
        xi = xs[s]
        yi = ys[s]
        for m in range(xgrid.size):
            dx = xi - xgrid[m]
            V[k, m] += A_el * yi / (dx*dx + yi*yi + rcore2)
    return V


def pinning_landscape(xs, ys, zs, p=P):
    xgrid = np.arange(p.x_lo, p.x_hi + 1e-9, p.dx_grid)
    V = _accumulate_V(xs, ys, zs, xgrid, p.Nz, p.A_el,
                      r_reg(p)**2, p.U_cap)
    Fpin = -np.gradient(V, p.dx_grid, axis=1)
    return xgrid, np.ascontiguousarray(V), np.ascontiguousarray(Fpin)


# =========================================== STAGE C : line dynamics ========
@njit(inline="always")
def _interp(F, k, x, x_lo, inv_dx, M):
    g = (x - x_lo) * inv_dx
    if g < 0.0 or g >= M - 1:
        return 0.0
    m = int(g)
    w = g - m
    return F[k, m]*(1.0 - w) + F[k, m + 1]*w


@njit(cache=True, fastmath=True)
def relax(x, Fpin, x_lo, dx_grid, kG, drive, Bdrag, dt,
          n_relax, tol_F, d_runaway, v_stall, chunk):
    """
    Zero-noise overdamped relaxation to mechanical equilibrium at fixed stress.

    The pinning criterion is the EXISTENCE OF AN EQUILIBRIUM, not a travelled
    distance: a line in a random landscape legitimately advances by finite
    amounts as the stress is raised, and re-pins each time.  So we declare
    "still pinned" when the line stalls (max force below tolerance, or mean
    velocity below v_stall) and "depinned" when it keeps moving past
    d_runaway measured from the start of THIS stress step.

    Returns  0 : equilibrium found -- still pinned
             1 : unbounded motion  -- depinned
    """
    Nz = x.size
    M = Fpin.shape[1]
    inv_dx = 1.0/dx_grid
    xn = np.empty(Nz)

    x_ref = 0.0
    for k in range(Nz):
        x_ref += x[k]
    x_ref /= Nz
    x_prev = x_ref

    for it in range(n_relax):
        fmax = 0.0
        s = 0.0
        for k in range(Nz):
            km = k - 1 if k > 0 else Nz - 1
            kp = k + 1 if k < Nz - 1 else 0
            fp = _interp(Fpin, k, x[k], x_lo, inv_dx, M)
            f = kG*(x[kp] - 2.0*x[k] + x[km]) + drive + fp
            if f > fmax:
                fmax = f
            elif -f > fmax:
                fmax = -f
            xn[k] = x[k] + dt*f/Bdrag
            s += xn[k]
        for k in range(Nz):
            x[k] = xn[k]
        xm = s/Nz

        if fmax < tol_F:
            return 0
        if xm - x_ref > d_runaway:
            return 1
        if (it + 1) % chunk == 0:
            if abs(xm - x_prev) < v_stall*chunk:
                return 0
            x_prev = xm
    return 1


@njit(cache=True, fastmath=True)
def first_passage(x, Fpin, x_lo, dx_grid, kG, drive, kT, Bdrag, dt,
                  max_steps, x_ref, d_runaway):
    """Langevin first-passage time (in steps) to escape at fixed stress."""
    Nz = x.size
    M = Fpin.shape[1]
    inv_dx = 1.0 / dx_grid
    amp = np.sqrt(2.0*kT*dt/Bdrag)
    xn = np.empty(Nz)

    for it in range(max_steps):
        s = 0.0
        for k in range(Nz):
            km = k - 1 if k > 0 else Nz - 1
            kp = k + 1 if k < Nz - 1 else 0
            fp = _interp(Fpin, k, x[k], x_lo, inv_dx, M)
            f = kG*(x[kp] - 2.0*x[k] + x[km]) + drive + fp
            xn[k] = x[k] + dt*f/Bdrag + amp*np.random.normal()
            s += xn[k]
        for k in range(Nz):
            x[k] = xn[k]
        if s/Nz - x_ref > d_runaway:
            return it
    return -1


def tau_c_quasistatic(Fpin, xgrid, x0, p=P, tau_max_gpa=8.0, keep_snaps=6):
    """
    Drag-free 0 K depinning stress: ramp tau quasi-statically, relaxing to
    force balance at every step, and return the first stress at which no
    equilibrium configuration exists.
    """
    kG = gamma_edge(p) / p.a
    x = np.full(p.Nz, x0)
    taus = np.linspace(0.0, tau_max_gpa*GPA, p.n_tau)

    snaps, snap_tau = [], []
    every = max(1, p.n_tau // 60)

    for i, tau in enumerate(taus):
        drive = tau * p.b * p.a
        code = relax(x, Fpin, xgrid[0], p.dx_grid, kG, drive, p.Bdrag, p.dt,
                     p.n_relax, p.tol_F, p.d_runaway, p.v_stall, p.chunk)
        if code == 1:
            return tau/GPA, np.array(snaps), np.array(snap_tau)/GPA
        if i % every == 0 and len(snaps) < keep_snaps:
            snaps.append(x.copy()); snap_tau.append(tau)
    return np.nan, np.array(snaps), np.array(snap_tau)/GPA


# ==================================== analytic: string-nucleation barrier ====
def string_barrier(xgrid, Vbar, tau, p=P, i_start=None):
    """
    dE*(tau) = 2 sqrt(2 Gamma) Int_{x1}^{x2} sqrt(Phi(x) - Phi(x1)) dx

    the exact saddle point of  H = Int [ Gamma/2 (dx/dz)^2 + Phi(x) ] dz ,
    with Phi = Vbar/a - tau b x the energy per unit length.

    IMPORTANT: x1 must be the METASTABLE local minimum the line actually sits
    in, not the global minimum of Phi.  Under an applied stress the tilt term
    -tau b x makes the global minimum run off to the right edge of the grid,
    which is a boundary artefact.  We therefore locate x1 by discrete gradient
    descent starting from the zero-stress trough.

    Returns dE* in eV; 0.0 once the barrier has vanished.
    """
    G = gamma_edge(p)
    Phi = Vbar/p.a - tau*p.b*xgrid
    n = Phi.size

    if i_start is None:
        i_start = int(np.argmin(Vbar))

    # --- descend to the local minimum
    i1 = int(np.clip(i_start, 1, n - 2))
    while 0 < i1 < n - 1:
        if Phi[i1 - 1] < Phi[i1]:
            i1 -= 1
        elif Phi[i1 + 1] < Phi[i1]:
            i1 += 1
        else:
            break
    if i1 >= n - 3 or i1 <= 1:
        return 0.0          # slid to a boundary: no metastable state left

    # --- climb to the barrier top
    i2 = i1
    while i2 < n - 1 and Phi[i2 + 1] > Phi[i2]:
        i2 += 1
    if i2 >= n - 2 or i2 == i1:
        return 0.0

    # --- descend to the turning point where Phi returns to Phi(x1)
    itp = i2
    while itp < n - 1 and Phi[itp] > Phi[i1]:
        itp += 1

    seg = slice(i1, max(itp, i1 + 2))
    integ = np.sqrt(np.clip(Phi[seg] - Phi[i1], 0.0, None))
    return 2.0*np.sqrt(2.0*G)*float(np.trapezoid(integ, xgrid[seg]))


def coherent_part(Vbar, p=P, sigma_A=3.0):
    """
    Split the z-averaged potential into a smooth COHERENT trough and a
    FLUCTUATION residual.  This is the same mean/variance decomposition that
    separates the two strengthening channels:

        tau_coh  <- smooth trough,  handled by string nucleation
        tau_VC   <- residual,       handled by Varvenne-Curtin statistics

    Without the split, noise-induced local minima in Vbar survive to arbitrary
    stress and corrupt the barrier bisection.
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(Vbar, sigma_A/p.dx_grid, mode="nearest")


def tau_c_coherent(xgrid, Vbar, p=P, depth_min=0.02):
    """
    Breakaway stress of a RIGID line from the coherent trough.

    At tau = 0 the well is open (Phi -> 0 as x -> inf) so the string barrier is
    formally infinite; the well-defined quantity is the maximum restoring
    force per unit length, phi = Vs/a:

        tau_coh = max_{x > x_min} (dphi/dx) / b

    Returns NaN when there is no coherent trough (random alloy).
    """
    Vs = coherent_part(Vbar, p)
    phi = Vs/p.a
    if phi.min() > -depth_min:
        return np.nan
    i0 = int(np.argmin(phi))
    g = np.gradient(phi, p.dx_grid)
    return float(g[i0:].max())/p.b/GPA


# ======================================================== VC reference =======
def varvenne_curtin(c, dv, p=P, alpha=1.0/12.0):
    """Varvenne-Curtin random-alloy 0 K strength and characteristic barrier."""
    pref = (1.0 + p.nu)/(1.0 - p.nu)
    S = c*dv**2/p.b**6
    dEb = 0.51*alpha**(1/3)*p.mu*p.b**3*pref**(2/3)*S**(1/3)
    ty0 = 0.040*alpha**(-1/3)*p.mu*pref**(4/3)*S**(2/3)
    return ty0/GPA, dEb


# ============================================================== driver =======
def main(p=P):
    G = gamma_edge(p)
    print(f"Gamma_edge  = {G:.3f} eV/A   (naive mu b^2/2 = {0.5*p.mu*p.b**2:.3f})")
    print(f"dt bound    = {p.Bdrag*p.a/(2*G):.2f}   (using dt = {p.dt})")
    # Match VC's line-tension parameter to the Dewit-Koehler Gamma actually
    # used here.  VC's default alpha = 1/12 is 3.2x softer than Gamma_edge for
    # Ta, and tau_y0 ~ alpha^(-1/3), so the default value is not a valid
    # reference for this simulation.
    alpha_eff = G/(p.mu*p.b**2)
    ty_vc, dEb_vc = varvenne_curtin(p.c0, 11.45, p, alpha=alpha_eff)
    ty_vc_def, _ = varvenne_curtin(p.c0, 11.45, p)
    tau_theo = p.mu/30.0/GPA
    print(f"VC random alloy: tau_y0 = {ty_vc:.3f} GPa (alpha = Gamma/mu b^2 = "
          f"{alpha_eff:.3f}), dE_b = {dEb_vc:.3f} eV")
    print(f"  [VC with its default alpha=1/12 would give {ty_vc_def:.3f} GPa; "
          f"not a like-for-like reference]")
    print(f"theoretical shear strength mu/30 = {tau_theo:.2f} GPa\n")

    thetas = [0.0, 0.02, 0.05, 0.10, 0.25, 1.00]
    n_seed = 6
    out = {}
    print(f"{'theta':>6} {'N_sol':>7} {'depth*':>9} {'tau_c(sim)':>16} "
          f"{'tau_c(coh)':>11} {'dE*(tc/2)':>10}")
    print("-" * 68)

    for th in thetas:
        tcs = []
        for sd in range(n_seed):
            rng = np.random.default_rng(p.seed + 977*sd + int(1000*th))
            xs, ys, zs, occ2d, X, Y = build_atmosphere(th, p, rng)
            xgrid, V, Fpin = pinning_landscape(xs, ys, zs, p)
            Vbar = V.mean(axis=0)
            x0 = float(xgrid[np.argmin(Vbar)])
            tc, snaps, snap_tau = tau_c_quasistatic(Fpin, xgrid, x0, p)
            tcs.append(tc)
            if sd == 0:                       # keep the first realisation
                tcoh = tau_c_coherent(xgrid, Vbar, p)
                Vs_ = coherent_part(Vbar, p)
                dE0 = (string_barrier(xgrid, Vs_, 0.5*tcoh*GPA, p,
                                      int(np.argmin(Vs_)))
                       if np.isfinite(tcoh) else np.nan)
                out[th] = dict(tau_coh=tcoh, dE0=dE0, xgrid=xgrid, Vbar=Vbar,
                               V=V, Fpin=Fpin, snaps=snaps, snap_tau=snap_tau,
                               x0=x0, xs=xs, ys=ys, nsol=xs.size,
                               depth=Vbar.min()/p.a)
        tcs = np.array(tcs)
        out[th]["tau_c"] = float(tcs.mean())
        out[th]["tau_sd"] = float(tcs.std())
        out[th]["tau_c0"] = float(tcs[0])
        print(f"{th:6.2f} {out[th]['nsol']:7d} {out[th]['depth']:9.3f} "
              f"{tcs.mean():8.3f} +/-{tcs.std():5.3f} "
              f"{out[th]['tau_coh']:11.3f} {out[th]['dE0']:9.2f}")

    print(f"\n * trough depth in eV/A (per unit dislocation length); "
          f"tau_c averaged over {n_seed} solute realisations")

    # --- finite-T first passage: test the (1 - tau/tau_c)^{3/2} scaling
    th_fp = 0.05
    r = out[th_fp]
    T_fp, dt_fp, d_esc, n_max = 220.0, 0.30, 12.0, 500_000
    print(f"\nfinite-T first passage: theta = {th_fp}, T = {T_fp:.0f} K, "
          f"tau_c(0 K) = {r['tau_c0']:.3f} GPa")
    fracs = np.array([0.72, 0.79, 0.85, 0.90, 0.95])
    kG = G/p.a
    fp_steps = []
    for f in fracs:
        tau = f*r["tau_c0"]*GPA
        ts = []
        for s in range(6):
            np.random.seed(p.seed + 13*s + int(1000*f))
            x = np.full(p.Nz, r["x0"])
            it = first_passage(x, r["Fpin"], r["xgrid"][0], p.dx_grid, kG,
                               tau*p.b*p.a, KB*T_fp, p.Bdrag, dt_fp,
                               n_max, r["x0"], d_esc)
            ts.append(it if it > 0 else n_max)
        med = float(np.median(ts))
        fp_steps.append(med)
        print(f"  tau/tau_c = {f:.2f}   median escape = {med:9.0f} steps"
              f"{'  (censored)' if med >= n_max else ''}")

    make_figures(p, thetas, out, fracs, np.array(fp_steps), th_fp,
                 ty_vc, tau_theo, T_fp)
    return out


# ================================================================ plots ======
def make_figures(p, thetas, out, fracs, fp_steps, th_fp, ty_vc, tau_theo, T_fp):
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # (a) occupancy cross-section
    occ1, _, X, Y = occupancy_profile(1.0, p)
    im = ax[0, 0].imshow(occ1.T, origin="lower", cmap="magma", vmin=0, vmax=1,
                         extent=[X.min(), X.max(), Y.min(), Y.max()])
    for th, col in [(0.05, "cyan"), (0.25, "lime")]:
        occ, _, _, _ = occupancy_profile(th, p)
        lev = 0.5*(occ.max() + p.c0)
        ax[0, 0].contour(X, Y, occ.T, levels=[lev], colors=[col], linewidths=1.6)
    ax[0, 0].axhline(0, color="w", lw=0.8)
    ax[0, 0].set_xlim(-40, 40); ax[0, 0].set_ylim(-40, 40)
    ax[0, 0].set_xlabel(r"$x$ [$\rm\AA$]"); ax[0, 0].set_ylabel(r"$y$ [$\rm\AA$]")
    ax[0, 0].set_title(r"(a) occupancy $c(x,y)$ at $\theta=1$" "\n"
                       r"cyan / green: half-saturation contour, $\theta$=0.05 / 0.25")
    fig.colorbar(im, ax=ax[0, 0], shrink=0.8)

    # (b) pinning landscape V(x,z)
    rb = out[0.25]
    r = out[th_fp]
    im = ax[0, 1].imshow(rb["V"], aspect="auto", origin="lower", cmap="viridis",
                         extent=[rb["xgrid"][0], rb["xgrid"][-1], 0, p.Nz*p.a])
    ax[0, 1].set_xlim(-40, 40)
    ax[0, 1].set_xlabel(r"dislocation position $x$ [$\rm\AA$]")
    ax[0, 1].set_ylabel(r"$z$ [$\rm\AA$]")
    ax[0, 1].set_title(r"(b) $\mathcal{V}(x,z)$ [eV/slice], $\theta$=0.25")
    fig.colorbar(im, ax=ax[0, 1], shrink=0.8)

    # (c) z-averaged troughs
    for th in [0.0, 0.05, 0.25, 1.0]:
        rr = out[th]
        ax[0, 2].plot(rr["xgrid"], rr["Vbar"]/p.a, lw=0.7, alpha=0.35, color="0.4")
        ax[0, 2].plot(rr["xgrid"], coherent_part(rr["Vbar"], p)/p.a, lw=2.0,
                      label=rf"$\theta$={th}")
    ax[0, 2].set_xlim(-40, 40)
    ax[0, 2].set_xlabel(r"$x$ [$\rm\AA$]")
    ax[0, 2].set_ylabel(r"$\bar{\mathcal{V}}/a$  [eV/$\rm\AA$]")
    ax[0, 2].set_title("(c) coherent trough (thick) vs. raw\n"
                   r"$z$-average (thin grey)")
    ax[0, 2].legend(fontsize=9); ax[0, 2].grid(alpha=0.3)

    # (d) quasi-static bow-out snapshots
    zz = np.arange(p.Nz)*p.a
    sn, st = r["snaps"], r["snap_tau"]
    for i in range(len(sn)):
        ax[1, 0].plot(sn[i] - r["x0"], zz, lw=1.4, label=rf"$\tau$={st[i]:.2f} GPa")
    ax[1, 0].set_xlabel(r"advance $x(z)-x_0$ [$\rm\AA$]")
    ax[1, 0].set_ylabel(r"$z$ [$\rm\AA$]")
    ax[1, 0].set_title(rf"(d) quasi-static bow-out, $\theta$={th_fp}"
                       "\n(0 K, force-balanced at every stress)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (e) tau_c vs ageing fraction
    th_arr = np.array(thetas)
    tc_sim = np.array([out[t]["tau_c"] for t in thetas])
    tc_sd = np.array([out[t]["tau_sd"] for t in thetas])
    tc_coh = np.array([out[t]["tau_coh"] for t in thetas])
    ax[1, 1].errorbar(th_arr, tc_sim, yerr=tc_sd, fmt="o-", lw=2, capsize=4,
                      label="3D flexible line (sim)")
    ax[1, 1].plot(th_arr, tc_coh, "s--", lw=1.6, color="tab:red",
                  label="coherent trough, rigid line")
    ax[1, 1].axhline(ty_vc, color="tab:blue", ls=":", lw=1.6,
                     label=rf"VC random alloy = {ty_vc:.2f} GPa")
    ax[1, 1].axhspan(tau_theo, 1e3, color="0.88", zorder=0)
    ax[1, 1].axhline(tau_theo, color="0.35", ls="-.", lw=1.4,
                     label=r"$\mu/30$: grey band is unattainable")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylim(0.05, 60)
    ax[1, 1].set_xlabel(r"ageing fraction $\theta$")
    ax[1, 1].set_ylabel(r"$\tau_c$ (0 K) [GPa]")
    ax[1, 1].set_title("(e) strength vs. degree of ageing")
    ax[1, 1].legend(fontsize=8, loc="lower right"); ax[1, 1].grid(alpha=0.3)

    # (f) first passage: test of the 3/2 scaling
    ok = fp_steps < 4.99e5
    ax[1, 2].plot((1.0 - fracs[ok])**1.5, np.log(fp_steps[ok]), "o", ms=9,
                  color="tab:red")
    if ok.sum() >= 2:
        xx = (1.0 - fracs[ok])**1.5
        cf = np.polyfit(xx, np.log(fp_steps[ok]), 1)
        xs_ = np.linspace(0, xx.max()*1.15, 50)
        ax[1, 2].plot(xs_, np.polyval(cf, xs_), "k--", lw=1.5,
                      label=rf"slope = {cf[0]:.1f} $= \Delta E_0/k_BT$"
                            f"\n$\\Rightarrow \\Delta E_0$ = {cf[0]*KB*T_fp:.2f} eV")
        ax[1, 2].legend(fontsize=9)
    ax[1, 2].set_xlabel(r"$(1-\tau/\tau_c)^{3/2}$")
    ax[1, 2].set_ylabel(r"$\ln\, t_{\rm escape}$")
    ax[1, 2].set_title(rf"(f) activation scaling, $T$={T_fp:.0f} K, $\theta$={th_fp}"
                       "\n" r"linearity confirms $\Delta E^*\propto(1-\tau/\tau_c)^{3/2}$")
    ax[1, 2].grid(alpha=0.3)

    fig.suptitle("3D flexible edge dislocation depinning from a frozen Cottrell "
                 "atmosphere in Ta(Lu)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig("depin3d_main.png", dpi=140)
    print("\nfigure written: depin3d_main.png")

    # --- separate: analytic barrier curve
    rr = out[0.05]
    tcoh = rr["tau_coh"]
    taus = np.linspace(0.05*tcoh, tcoh*0.995, 60)
    Vs = coherent_part(rr["Vbar"], p)
    i0 = int(np.argmin(Vs))
    dEs = np.array([string_barrier(rr["xgrid"], Vs, t*GPA, p, i0)
                    for t in taus])
    plt.figure(figsize=(6.6, 4.7))
    plt.plot(taus, dEs, lw=2.2, color="tab:red", label="exact saddle point")
    plt.plot(taus, dEs[0]*((1 - taus/tcoh)/(1 - taus[0]/tcoh))**1.5, "k--", lw=1.5,
             label=r"$\Delta E_0(1-\tau/\tau_c)^{3/2}$")
    plt.xlabel(r"$\tau$ [GPa]"); plt.ylabel(r"$\Delta E^*$ [eV]")
    plt.title(r"string-nucleation barrier of the coherent trough, $\theta$=0.05")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("depin3d_barrier.png", dpi=140)
    print("figure written: depin3d_barrier.png")


if __name__ == "__main__":
    np.random.seed(P.seed)
    main()
