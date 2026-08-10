"""
Kink-pair kinetic Monte Carlo for a 1/2<111> screw dislocation in Ta(Lu).

WHY depin3d.py's ALGORITHM DOES NOT APPLY HERE
    That code relaxes a continuous string in a smooth landscape.  A bcc screw
    does not glide that way: it sits in deep Peierls valleys and advances by
    nucleating kink pairs and migrating kinks.  The continuous functional

        H[x(z)] = Int dz [ Gamma/2 (dx/dz)^2 + V_P(x) + V_sol(x,z) - tau b x ]

    reduces, for k_B T << E_k, to a solid-on-solid model in the integer valley
    index n_k:

        H = E_k sum_k |n_k - n_{k+1}| + sum_k W[n_k, k] - tau b h a_z sum_k n_k

    V_P has been absorbed into the kink energy E_k = (2h/pi) sqrt(2 Gamma V_0)
    and the kink migration barrier E_km.  Nucleation and migration then need
    no special-casing: flipping a site on a flat segment costs 2 E_k
    automatically, flipping one next to a kink costs zero kink energy.

ALGORITHM
    Rejection-free BKL / Gillespie.  Every site has two events (up, down) with

        r = nu_D exp( -[E_km + max(0, dE)] / k_B T )

    which satisfies detailed balance.  After an event only sites k-1, k, k+1
    change rate, so each step is O(1) plus the search for the selected event.

WHAT IT IS FOR
    Solid solution softening: disorder lowers the EFFECTIVE nucleation
    enthalpy by sigma^2/(2 k_B T) because nucleation samples the low tail of
    the barrier distribution.  That term grows as T falls, so a random solute
    field SOFTENS a bcc metal at low temperature while hardening it at high
    temperature, where kink migration drag takes over.  This script measures
    the crossover directly.

    The solute field strength U_cap is a PLACEHOLDER until the screw-core DFT
    (screw_dft_cells.py) is done.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

import stroh_field as SF
from depin3d import GPA, KB

# --- screw geometry and elastic dipole (inlined; importing screw_depin would
#     execute its module-level driver)
C11, C12, C44 = 266.0, 158.2, 87.4
K_BULK = (C11 + 2*C12)/3.0
OMEGA_REL = 11.45
PDIP = K_BULK*GPA*OMEGA_REL


def screw_frame():
    e1 = np.array([1., -1., 0.])/np.sqrt(2)
    e3 = np.array([1., 1., 1.])/np.sqrt(3)
    e2 = np.cross(e3, e1); e2 /= np.linalg.norm(e2)
    return np.vstack([e1, e2, e3])

NU_D = 1.0e13            # s^-1, attempt frequency


class K:
    # --- crystal
    a       = 2.863      # A, lattice spacing used for both the solute grid
    b       = 2.863      # A, |b|
    h       = 2.863      # A, Peierls valley spacing (commensurate with a:
                         #    the real {110} spacing is a0*sqrt(6)/3 = 2.699 A;
                         #    forcing commensuration is the standard SOS
                         #    simplification and shifts E_k by ~5 %)
    # --- dislocation
    Gamma   = 3.935      # eV/A, Dewit-Koehler screw line tension
    tau_P   = 0.90       # GPa, Peierls stress of the Ta screw
    E_k     = 0.45       # eV, single-kink energy (literature; see check below)
    E_km    = 0.02       # eV, kink migration barrier
    E_kk    = 0.61       # eV*segment, kink-kink attraction prefactor:
                         # E_int(w) = E_kk/w, from mu b^2 h^2/[8 pi (1-nu) a_z]
    w_max   = 16         # widest kink pair considered

    # --- solute field
    c0      = 0.02
    U_cap   = 0.30       # eV, PLACEHOLDER for the screw-core binding energy
    Nx      = 64         # valleys (periodic in x)
    Ny      = 48
    Nz      = 96         # segments along the line
    R_keep  = 45.0       # A
    r_chem  = 1.6        # in units of a: radius of the "core region" in which
                         # a Lu atom is taken to chemically alter the Peierls
                         # barrier

    # --- KMC
    n_steps = 150_000
    n_therm = 20_000
    seed    = 20260807


def kink_energy_check(p=K):
    """E_k from the sine-Gordon soliton, for consistency with tau_P."""
    tp = p.tau_P*GPA
    V0 = tp*p.b*p.h/np.pi                  # eV/A
    Ek = (2*p.h/np.pi)*np.sqrt(2*p.Gamma*V0)
    return Ek, V0


# ------------------------------------------------------- solute landscape
def solute_landscape(p=K, seed=0, c0=None):
    """
    W[n, k] = sum over solutes in slice k of U_screw(x_i - n h, y_i),
    with the minimum image convention in x so the line can glide indefinitely.
    """
    c0 = p.c0 if c0 is None else c0
    if c0 <= 0:
        return (np.zeros((p.Nx, p.Nz)),
                np.zeros((p.Nx, p.Nz), dtype=np.int64))

    st = SF.Stroh(SF.rotate_C(
        SF.cubic_C(C11*GPA, C12*GPA, C44*GPA), screw_frame()))
    Pdip = PDIP
    bvec = np.array([0.0, 0.0, p.b])
    Lx = p.Nx*p.h

    rng = np.random.default_rng(seed)
    ix = np.arange(p.Nx)*p.h
    iy = (np.arange(p.Ny) - p.Ny//2)*p.a
    X, Y = np.meshgrid(ix, iy, indexing="ij")
    keep = (np.minimum(X, Lx - X)**2 + Y**2) <= p.R_keep**2

    # random solid solution: screws build no atmosphere (screw_depin.py)
    occ = rng.random((p.Nx, p.Ny, p.Nz)) < c0
    occ &= keep[:, :, None]

    W = np.zeros((p.Nx, p.Nz))
    I = np.zeros((p.Nx, p.Nz), dtype=np.int64)      # Lu in the core region?
    xi, yi, zi = np.nonzero(occ)
    sx = ix[xi]; sy = iy[yi]
    for n in range(p.Nx):
        dx = sx - n*p.h
        dx -= Lx*np.round(dx/Lx)                    # minimum image
        u = SF.interaction_field(dx, sy, Pdip, st, bvec, p.U_cap)
        np.add.at(W[n], zi, u)
        core = (dx*dx + sy*sy) < (p.r_chem*p.a)**2
        if core.any():
            np.add.at(I[n], zi[core], 1)
    return np.ascontiguousarray(W), np.ascontiguousarray(I)


# ------------------------------------------------------------- KMC kernel
@njit(cache=True, fastmath=True)
def _Eint(w, Ekk):
    """Elastic attraction of two opposite kinks separated by w segments.

    Without this term the SOS model gives a nucleation barrier of exactly
    2 E_k at every width, there is no critical nucleus, and the glide velocity
    at 100 K comes out as 1e-27 A/s.  The attraction makes narrow pairs cheap,
    so the barrier is a maximum at a finite w* that shrinks with stress.
    """
    return Ekk/w


@njit(cache=True, fastmath=True)
def _nucleation(n, k, W, I, kappa, Ek, Ekk, work, Nx, Nz, w_max):
    """
    Barrier and optimal width of a kink pair nucleated at site k:

        dH(w) = 2 E_k - E_int(w) - w*work + dE_elastic(w) - kappa * I(Lu)

    I(Lu) = 1 if any site swept by the nucleus contains a Lu atom in the core
    region.  kappa is the phenomenological CHEMICAL softening: the direct
    reduction of the Peierls barrier by the solute's electronic structure,
    which the elastic term dE_elastic cannot produce.  It is what the NEB
    calculation (neb_kink_inputs.py) is meant to supply.
    """
    nk = n[k]
    best = -1.0e30
    w_best = 1
    dW = 0.0
    has_lu = 0
    for w in range(1, w_max+1):
        j = k + w - 1
        if j >= Nz:
            break
        if n[j] != nk:
            break
        dW += W[(nk+1) % Nx, j] - W[nk % Nx, j]
        if I[(nk+1) % Nx, j] > 0:
            has_lu = 1
        h = 2.0*Ek - _Eint(w, Ekk) - w*work + dW - kappa*has_lu
        if h > best:
            best = h
            w_best = w
    if best < 0.0:
        best = 0.0
    return best, w_best


@njit(cache=True, fastmath=True)
def _dE(n, k, dn, W, Ek, work, Nx, Nz):
    """Energy change of moving segment k by dn valleys."""
    km = k - 1 if k > 0 else Nz - 1
    kp = k + 1 if k < Nz - 1 else 0
    nk = n[k]; nn = nk + dn
    e = Ek*(abs(nn - n[km]) + abs(nn - n[kp])
            - abs(nk - n[km]) - abs(nk - n[kp]))
    e += W[nn % Nx, k] - W[nk % Nx, k]
    e -= dn*work
    return e


@njit(cache=True, fastmath=True)
def _has_kink(n, k, Nz):
    km = k - 1 if k > 0 else Nz - 1
    kp = k + 1 if k < Nz - 1 else 0
    return n[k] != n[km] or n[k] != n[kp]


@njit(cache=True, fastmath=True)
def _rates_at(n, k, W, I, kappa, Ek, Ekk, Ekm, work, beta, Nx, Nz, w_max, R):
    """R[k,0] nucleation, R[k,1] up-migration, R[k,2] down-migration."""
    if _has_kink(n, k, Nz):
        R[k, 0] = 0.0
        for d in (1, -1):
            e = _dE(n, k, d, W, Ek, work, Nx, Nz)
            if e < 0.0:
                e = 0.0
            R[k, 1 if d == 1 else 2] = np.exp(-beta*(Ekm + e))
    else:
        h, _ = _nucleation(n, k, W, I, kappa, Ek, Ekk, work, Nx, Nz, w_max)
        R[k, 0] = np.exp(-beta*h)
        R[k, 1] = 0.0
        R[k, 2] = 0.0


@njit(cache=True, fastmath=True)
def kmc_run(W, I, kappa, Ek, Ekk, Ekm, work, beta, Nz, Nx, n_steps, n_therm, nu, w_max):
    """Rejection-free BKL over three event classes per site."""
    n = np.zeros(Nz, dtype=np.int64)
    R = np.zeros((Nz, 3))
    for k in range(Nz):
        _rates_at(n, k, W, I, kappa, Ek, Ekk, Ekm, work, beta, Nx, Nz, w_max, R)

    t = 0.0
    t0 = 0.0
    n_mean0 = 0.0
    kink_acc = 0.0
    n_samp = 0

    for step in range(n_steps):
        tot = 0.0
        for k in range(Nz):
            tot += R[k, 0] + R[k, 1] + R[k, 2]
        if tot <= 0.0:
            break
        t += -np.log(np.random.random())/(tot*nu)

        target = np.random.random()*tot
        acc = 0.0
        sel_k = Nz - 1
        sel_e = 2
        for k in range(Nz):
            for e in range(3):
                acc += R[k, e]
                if acc > target:
                    sel_k = k
                    sel_e = e
                    break
            if acc > target:
                break

        if sel_e == 0:
            _, w = _nucleation(n, sel_k, W, I, kappa, Ek, Ekk, work, Nx, Nz, w_max)
            for j in range(sel_k, min(sel_k+w, Nz)):
                n[j] += 1
            lo = sel_k - 1
            hi = sel_k + w
        else:
            n[sel_k] += 1 if sel_e == 1 else -1
            lo = sel_k - 1
            hi = sel_k + 1

        # a nucleation changes the local flatness, so refresh a window that
        # also covers every site whose optimal width could have changed
        for k in range(max(0, lo-w_max), min(Nz, hi+w_max+1)):
            _rates_at(n, k, W, I, kappa, Ek, Ekk, Ekm, work, beta, Nx, Nz, w_max, R)

        if step == n_therm:
            t0 = t
            s = 0.0
            for k in range(Nz):
                s += n[k]
            n_mean0 = s/Nz
        if step > n_therm and step % 500 == 0:
            kk = 0.0
            for k in range(Nz-1):
                kk += abs(n[k+1] - n[k])
            kink_acc += kk/Nz
            n_samp += 1

    s = 0.0
    for k in range(Nz):
        s += n[k]
    dn = s/Nz - n_mean0
    dt_tot = t - t0
    v = dn/dt_tot if dt_tot > 0 else 0.0
    kd = kink_acc/n_samp if n_samp > 0 else 0.0
    return v, kd


def velocity(tau_gpa, T, WI, p=K, seed=1, kappa=0.0):
    """Dislocation glide velocity in A/s."""
    beta = 1.0/(KB*T)
    work = tau_gpa*GPA*p.b*p.h*p.a          # eV gained per valley per segment
    np.random.seed(seed)
    W, I = WI
    v_valleys, kd = kmc_run(W, I, kappa, p.E_k, p.E_kk, p.E_km, work, beta,
                            p.Nz, p.Nx, p.n_steps, p.n_therm, NU_D, p.w_max)
    return v_valleys*p.h, kd


def tau_for_velocity(T, W, v_target, p=K, lo=1e-4, hi=1.3, n_bis=13, seed=1, kappa=0.0):
    """
    Bisect in LOG stress for the threshold at the target velocity.

    Linear bisection over a narrow bracket silently saturated at both ends:
    pure Ta above 200 K pinned at the lower bound and the alloy at 100 K at
    the upper bound, so the alloy/pure ratio was meaningless.  Returns NaN
    when the answer lies outside the bracket instead of reporting the bound.
    """
    v_lo, _ = velocity(lo, T, W, p, seed, kappa)
    v_hi, _ = velocity(hi, T, W, p, seed, kappa)
    if v_lo > v_target:
        return np.nan          # flows even at essentially zero stress
    if v_hi < v_target:
        return np.nan          # athermal: cannot reach the rate below tau_P
    a, b_ = np.log(lo), np.log(hi)
    for _ in range(n_bis):
        m = 0.5*(a + b_)
        v, _ = velocity(np.exp(m), T, W, p, seed, kappa)
        if v < v_target:
            a = m
        else:
            b_ = m
    return float(np.exp(0.5*(a + b_)))


# -------------------------------------------------------------------- main
def main(p=K):
    Ek_sg, V0 = kink_energy_check(p)
    print(f"sine-Gordon consistency: E_k from tau_P and Gamma = {Ek_sg:.3f} eV")
    print(f"  (using E_k = {p.E_k:.3f} eV from the literature; V_0 = "
          f"{V0:.4f} eV/A, tau_P = {p.tau_P} GPa)")

    # target velocity for gdot = 1e-3 /s at rho_m = 1e12 m^-2
    rho, gdot = 1e12, 1e-3
    v_target = gdot/(rho*p.b*1e-10)*1e10          # A/s
    print(f"target velocity for gdot = {gdot:.0e}/s, rho = {rho:.0e} m^-2: "
          f"v = {v_target:.3e} A/s\n")

    W0 = solute_landscape(p, seed=p.seed, c0=0.0)
    Wc = solute_landscape(p, seed=p.seed)
    sig = Wc.std()
    print(f"solute landscape: c0 = {p.c0}, U_cap = {p.U_cap} eV")
    print(f"  sd of W over valleys/slices = {sig:.4f} eV per segment")
    print(f"  softening scale sigma^2/(2kT):")
    for T in [50, 100, 200, 300]:
        print(f"    T = {T:3d} K -> {sig**2/(2*KB*T)*1000:6.1f} meV")
    print()

    temps = [100, 150, 200, 300, 450]
    print(f"{'T [K]':>6} {'tau_pure':>10} {'tau_alloy':>10} {'ratio':>8} "
          f"{'kink dens':>10}")
    print("-"*50)
    out = {"T": temps, "pure": [], "alloy": []}
    for T in temps:
        tp = np.nanmean([tau_for_velocity(T, W0, v_target, p, seed=11+7*s)
                         for s in range(2)])
        ta = np.nanmean([tau_for_velocity(T, Wc, v_target, p, seed=11+7*s)
                         for s in range(2)])
        _, kd = velocity(ta if np.isfinite(ta) else 0.3, T, Wc, p, seed=11)
        out["pure"].append(tp); out["alloy"].append(ta)
        print(f"{T:6d} {tp:10.3f} {ta:10.3f} {ta/tp:8.3f} {kd:10.4f}")

    r = np.array(out["alloy"])/np.array(out["pure"])
    soft = [t for t, x in zip(temps, r) if np.isfinite(x) and x < 1.0]
    print()
    if soft:
        print(f"SOLID SOLUTION SOFTENING at T = {soft} K "
              f"(alloy weaker than pure)")
        print(f"  strongest softening: {100*(1-np.nanmin(r)):.1f} % at "
              f"T = {temps[int(np.nanargmin(r))]} K")
    else:
        print("no softening window found at these parameters")
    hard = [t for t, x in zip(temps, r) if np.isfinite(x) and x > 1.0]
    if hard:
        print(f"hardening at T = {hard} K")

    make_figure(out, sig, p)
    return out


def make_figure(out, sig, p):
    T = np.array(out["T"], float)
    tp = np.array(out["pure"]); ta = np.array(out["alloy"])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax[0].plot(T, tp, "o-", lw=2, label="pure Ta")
    ax[0].plot(T, ta, "s-", lw=2, label=rf"Ta + {100*p.c0:.0f}% Lu")
    ax[0].set_xlabel("T [K]"); ax[0].set_ylabel(r"$\tau_y$ [GPa]")
    ax[0].set_title("(a) flow stress at fixed strain rate")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    ax[1].plot(T, ta/tp, "o-", lw=2, color="tab:red")
    ax[1].axhline(1.0, color="k", lw=1.2)
    ax[1].fill_between(T, 0, 1, where=(ta/tp < 1), color="tab:green",
                       alpha=0.15)
    ax[1].set_xlabel("T [K]")
    ax[1].set_ylabel(r"$\tau_y^{\rm alloy}/\tau_y^{\rm pure}$")
    ax[1].set_title("(b) softening below 1, hardening above")
    ax[1].grid(alpha=0.3)

    Tf = np.linspace(30, 320, 200)
    ax[2].plot(Tf, 1000*sig**2/(2*KB*Tf), lw=2, color="tab:purple",
               label=r"annealed  $\sigma^2/2k_BT$")
    Nsite = p.Nz
    ax[2].axhline(1000*sig*np.sqrt(2*np.log(Nsite)), color="k", ls="--",
                  label=r"quenched limit $\sigma\sqrt{2\ln N}$")
    Tstar = sig/np.sqrt(2*np.log(Nsite))/KB
    ax[2].axvline(Tstar, color="0.5", ls=":",
                  label=rf"$T^*$ = {Tstar:.0f} K")
    ax[2].set_xlabel("T [K]")
    ax[2].set_ylabel(r"barrier reduction [meV]")
    ax[2].set_title("(c) why disorder softens")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("Kink-pair KMC: solid solution softening of the bcc screw "
                 "dislocation in Ta(Lu)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("kink_kmc.png", dpi=140)
    print("\nfigure written: kink_kmc.png")


if __name__ == "__main__":
    main()
