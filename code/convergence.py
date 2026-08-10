"""
Convergence tests for the 3D depinning model (depin3d.py).

Every number reported so far rests on four numerical parameters that were
chosen by hand and never tested:

    Nz        line length (periodic)          128 sites = 366 A
    R_keep    solute inclusion radius         60 A
    dx_grid   pinning-landscape resolution    0.20 A
    n_tau     stress-ramp resolution          500 steps over 8 GPa

plus the number of independent solute realisations needed for a stated error
bar.  This script sweeps each one around the reference point and reports the
converged value together with the discretisation error, so that downstream
results can be quoted to a defensible precision.

Results are checkpointed to convergence.json after every parameter, so an
interrupted run loses at most one sweep.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import depin3d as D
from depin3d import GPA

CKPT = "convergence.json"
THETA = 0.25          # partially aged: has both a coherent trough and
                      # fluctuations, so it exercises every part of the code


# ------------------------------------------------------------------ helpers
def mkP(**kw):
    """A copy of depin3d.P with selected attributes overridden."""
    class Q(D.P):
        pass
    for k, v in kw.items():
        setattr(Q, k, v)
    return Q


def run_one(p, theta, seed, tau_max=8.0):
    """Full pipeline -> tau_c in GPa."""
    rng = np.random.default_rng(seed)
    xs, ys, zs, occ, X, Y = D.build_atmosphere(theta, p, rng)
    xgrid, V, Fpin = D.pinning_landscape(xs, ys, zs, p)
    Vbar = V.mean(axis=0)
    x0 = float(xgrid[np.argmin(Vbar)])
    tc, _, _ = D.tau_c_quasistatic(Fpin, xgrid, x0, p, tau_max_gpa=tau_max)
    return tc


def sweep(name, values, mkkw, seeds, theta=THETA, tau_max=8.0):
    """Run a one-parameter sweep and return {value: (mean, sem, n)}."""
    out = {}
    for v in values:
        p = mkP(**mkkw(v))
        t0 = time.time()
        tcs = [run_one(p, theta, D.P.seed + 977*s + 13, tau_max)
               for s in range(seeds)]
        tcs = np.array(tcs, float)
        good = tcs[np.isfinite(tcs)]
        m = float(good.mean()) if good.size else np.nan
        sem = float(good.std(ddof=1)/np.sqrt(good.size)) if good.size > 1 else np.nan
        out[str(v)] = (m, sem, int(good.size))
        print(f"    {name} = {v!s:>6}   tau_c = {m:6.3f} +/- {sem:5.3f} GPa "
              f"(n={good.size})   [{time.time()-t0:5.1f} s]")
    return out


def save(res):
    with open(CKPT, "w") as f:
        json.dump(res, f, indent=2)


def load():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


# -------------------------------------------------------------------- main
def main(which=None):
    res = load()
    ref = D.P
    print(f"reference: Nz={ref.Nz}, R_keep={ref.R_keep}, "
          f"dx_grid={ref.dx_grid}, n_tau={ref.n_tau}, theta={THETA}\n")

    todo = which or ["dx_grid", "n_tau", "R_keep", "Nz", "seeds"]

    if "dx_grid" in todo and "dx_grid" not in res:
        print("[1/5] pinning-landscape resolution dx_grid")
        res["dx_grid"] = sweep("dx_grid", [0.05, 0.10, 0.20, 0.40, 0.80],
                               lambda v: dict(dx_grid=v), seeds=4)
        save(res)

    if "n_tau" in todo and "n_tau" not in res:
        print("\n[2/5] stress-ramp resolution n_tau  (over 0-8 GPa)")
        res["n_tau"] = sweep("n_tau", [125, 250, 500, 1000, 2000],
                             lambda v: dict(n_tau=v), seeds=4)
        save(res)

    if "R_keep" in todo and "R_keep" not in res:
        print("\n[3/5] solute inclusion radius R_keep")
        res["R_keep"] = sweep("R_keep", [20.0, 30.0, 40.0, 60.0, 85.0],
                              lambda v: dict(R_keep=v), seeds=4)
        save(res)

    if "Nz" in todo and "Nz" not in res:
        print("\n[4/5] line length Nz  (weakest-link statistics: expect a "
              "decrease then a plateau)")
        res["Nz"] = sweep("Nz", [32, 64, 128, 256, 512],
                          lambda v: dict(Nz=v), seeds=4)
        save(res)

    if "seeds" in todo and "seeds" not in res:
        print("\n[5/5] realisation-to-realisation scatter")
        p = mkP()
        tcs = []
        for s in range(24):
            tcs.append(run_one(p, THETA, D.P.seed + 977*s + 13))
        tcs = np.array(tcs, float)
        tcs = tcs[np.isfinite(tcs)]
        res["seeds"] = dict(values=tcs.tolist(),
                            mean=float(tcs.mean()),
                            std=float(tcs.std(ddof=1)))
        print(f"    n=24: mean = {tcs.mean():.3f} GPa, "
              f"sd = {tcs.std(ddof=1):.3f} GPa "
              f"({100*tcs.std(ddof=1)/tcs.mean():.1f} % relative)")
        save(res)

    report(res)
    return res


# ------------------------------------------------------------------ report
def report(res):
    print("\n" + "=" * 66)
    print("CONVERGENCE SUMMARY")
    print("=" * 66)

    ref = {"dx_grid": "0.2", "n_tau": "500", "R_keep": "60.0", "Nz": "128"}
    finest = {"dx_grid": "0.05", "n_tau": "2000", "R_keep": "85.0",
              "Nz": "512"}

    for k in ["dx_grid", "n_tau", "R_keep", "Nz"]:
        if k not in res:
            continue
        d = res[k]
        if ref[k] not in d or finest[k] not in d:
            continue
        m_ref, s_ref, _ = d[ref[k]]
        m_fin, s_fin, _ = d[finest[k]]
        err = 100*(m_ref - m_fin)/m_fin
        sig = abs(m_ref - m_fin) > 2*np.hypot(s_ref, s_fin)
        print(f"  {k:>8}: reference {m_ref:6.3f} vs finest {m_fin:6.3f} GPa"
              f"   -> {err:+6.1f} %"
              f"   {'SIGNIFICANT' if sig else 'within noise'}")

    if "seeds" in res:
        v = np.array(res["seeds"]["values"])
        sd = res["seeds"]["std"]
        m = res["seeds"]["mean"]
        print(f"\n  realisation scatter: sd/mean = {100*sd/m:.1f} %")
        for target in [10.0, 5.0, 2.0]:
            n = (100*sd/m/target)**2
            print(f"    to quote the mean to +/-{target:.0f} %: "
                  f"n >= {int(np.ceil(n))} realisations")
        print(f"\n  => with the 2-3 realisations used so far, tau_c is known "
              f"to about +/-{100*sd/m/np.sqrt(2.5):.0f} %.")
        print(f"     Quote it to 2 significant figures, not 3.")


def make_figure(res):
    keys = [k for k in ["dx_grid", "n_tau", "R_keep", "Nz"] if k in res]
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(4.2*(len(keys)+1), 4.0))
    labels = {"dx_grid": r"$dx_{\rm grid}$ [$\rm\AA$]",
              "n_tau": r"$n_\tau$ (0-8 GPa)",
              "R_keep": r"$R_{\rm keep}$ [$\rm\AA$]",
              "Nz": r"$N_z$ (line length, sites)"}
    for i, k in enumerate(keys):
        d = res[k]
        xs = np.array(sorted(float(v) for v in d))
        ys = np.array([d[fmt_key(k, x)][0] for x in xs])
        es = np.array([d[fmt_key(k, x)][1] for x in xs])
        ax[i].errorbar(xs, ys, yerr=es, fmt="o-", lw=1.8, capsize=4)
        ax[i].set_xscale("log")
        ax[i].set_xlabel(labels[k]); ax[i].set_ylabel(r"$\tau_c$ [GPa]")
        ax[i].set_title(f"({chr(97+i)}) {k}")
        ax[i].grid(alpha=0.3, which="both")

    if "seeds" in res:
        v = np.array(res["seeds"]["values"])
        run = np.array([v[:n+1].mean() for n in range(len(v))])
        sem = np.array([v[:n+1].std(ddof=1)/np.sqrt(n+1) if n > 0 else np.nan
                        for n in range(len(v))])
        n = np.arange(1, len(v)+1)
        a = ax[-1]
        a.plot(n, run, "o-", lw=1.8, label="running mean")
        a.fill_between(n, run-sem, run+sem, alpha=0.25, label=r"$\pm$ SEM")
        a.axhline(v.mean(), color="k", ls="--", lw=1.0)
        a.set_xlabel("number of solute realisations")
        a.set_ylabel(r"$\tau_c$ [GPa]")
        a.set_title(f"({chr(97+len(keys))}) realisation scatter")
        a.legend(fontsize=9); a.grid(alpha=0.3)

    fig.suptitle(r"Convergence of $\tau_c$ (0 K, $\theta$=0.25) with respect "
                 "to every numerical parameter", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("convergence.png", dpi=140)
    print("\nfigure written: convergence.png")


def fmt_key(k, x):
    if k in ("n_tau", "Nz"):
        return str(int(x))
    return str(x)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or None
    r = main(which)
    make_figure(r)
