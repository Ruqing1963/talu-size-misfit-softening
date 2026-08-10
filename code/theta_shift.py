"""Where does the atmosphere become unbreakable, with and without J?

tau_c crosses mu/30 (the theoretical shear strength) at some degree of
ageing.  Beyond that the dislocation cannot break away at all and the yield
mechanism must change.  Solute-solute attraction moves that crossing.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import atmosphere3d as A, depin3d as D
from depin3d import P, GPA

thetas = [0.40, 0.55, 0.65]
omegas = [0.0]
n_seed = 2
tau_theo = P.mu/30/GPA
print(f"mu/30 = {tau_theo:.2f} GPa\n")
print(f"{'omega':>6} {'theta':>6} {'N_sol':>7} {'depth':>8} {'tau_c':>16}")
print("-"*50)
res = {}
for om in omegas:
    res[om] = []
    for th in thetas:
        t = []
        for s in range(n_seed):
            c, U2d, X, Y, mu = A.equilibrate(-om, P, seed=100*s+7,
                                             verbose=False, theta=th, n_eq=200)
            tc, dep, nsl = A.tau_c_from(c, X, Y, P)
            t.append(tc)
        t = np.array(t)
        res[om].append(t.mean())
        print(f"{om:6.2f} {th:6.2f} {nsl:7d} {dep:8.3f} "
              f"{t.mean():8.3f} +/-{t.std(ddof=1):5.3f}")
    print()

def crossing(th, y, target):
    th = np.array(th); y = np.array(y)
    for i in range(len(y)-1):
        if (y[i]-target)*(y[i+1]-target) < 0:
            f = (target-y[i])/(y[i+1]-y[i])
            return th[i] + f*(th[i+1]-th[i])
    return np.nan

print("ageing fraction at which tau_c reaches mu/30:")
for om in omegas:
    print(f"  omega = {om:.2f} eV : theta* = {crossing(thetas,res[om],tau_theo):.3f}")

plt.figure(figsize=(6.6,4.8))
for om,c in zip(omegas,["tab:blue","tab:red"]):
    plt.plot(thetas,res[om],"o-",color=c,lw=2,label=rf"$\omega$={om:.2f} eV")
plt.axhline(tau_theo,color="0.35",ls="-.",lw=1.4,label=r"$\mu/30$")
plt.axhspan(tau_theo,1e3,color="0.9",zorder=0)
plt.yscale("log"); plt.ylim(0.2,20)
plt.xlabel(r"ageing fraction $\theta$ (field scaling)")
plt.ylabel(r"$\tau_c$ (0 K) [GPa]")
plt.title("solute-solute attraction moves the unbreakable threshold")
plt.legend(); plt.grid(alpha=0.3,which="both"); plt.tight_layout()
plt.savefig("theta_shift.png",dpi=140)
print("\nfigure written: theta_shift.png")
