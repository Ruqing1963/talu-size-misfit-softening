"""How large must the chemical softening kappa be to reverse the hardening?

The statistical-fluctuation mechanism alone gives hardening at every
temperature (kink_kmc.py).  This scan asks the falsifiable question instead:
what value of kappa -- the direct reduction of the kink-pair nucleation
barrier by a Lu atom in the core -- is needed before the alloy becomes weaker
than pure Ta?  That threshold is a prediction the NEB calculation can confirm
or refute.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kink_kmc as KK
from depin3d import KB
p = KK.K

rho, gdot = 1e12, 1e-3
v_t = gdot/(rho*p.b*1e-10)*1e10
W0 = KK.solute_landscape(p, seed=p.seed, c0=0.0)
Wc = KK.solute_landscape(p, seed=p.seed)
print(f"target v = {v_t:.2e} A/s;  2 E_k = {2*p.E_k:.3f} eV;  "
      f"Lu in core at {100*Wc[1].mean():.1f} % of (valley, slice) pairs")

kappas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
temps = [150, 250]
res = {}
for T in temps:
    tp = np.nanmean([KK.tau_for_velocity(T, W0, v_t, p, seed=11+7*s)
                     for s in range(2)])
    row = []
    for kap in kappas:
        ta = np.nanmean([KK.tau_for_velocity(T, Wc, v_t, p, seed=11+7*s,
                                             kappa=kap) for s in range(2)])
        row.append(ta)
        print(f"T={T:4d}  kappa={kap:.2f}  tau_pure={tp:.3f}  "
              f"tau_alloy={ta:.3f}  ratio={ta/tp:.3f}")
    res[T] = (tp, np.array(row))
    print()

print("kappa threshold for tau_alloy < tau_pure:")
for T in temps:
    tp, row = res[T]
    r = row/tp
    kt = np.nan
    for i in range(len(kappas)-1):
        if (r[i]-1)*(r[i+1]-1) < 0:
            f = (1-r[i])/(r[i+1]-r[i])
            kt = kappas[i] + f*(kappas[i+1]-kappas[i])
    print(f"  T = {T} K : kappa* = {kt:.3f} eV "
          f"= {100*kt/(2*p.E_k):.1f} % of the pure nucleation barrier")

plt.figure(figsize=(6.8,4.8))
for T,c in zip(temps,["tab:blue","tab:red"]):
    tp,row = res[T]
    plt.plot(kappas,row/tp,"o-",color=c,lw=2,label=f"T = {T} K")
plt.axhline(1.0,color="k",lw=1.3)
plt.fill_between([0,max(kappas)],0,1,color="tab:green",alpha=0.12)
plt.text(0.005,0.55,"softening",fontsize=10,color="tab:green")
plt.xlabel(r"chemical softening $\kappa$ [eV]")
plt.ylabel(r"$\tau_y^{\rm alloy}/\tau_y^{\rm pure}$")
plt.title(r"threshold $\kappa$ for solid solution softening")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("kappa_scan.png",dpi=140)
print("\nfigure written: kappa_scan.png")
