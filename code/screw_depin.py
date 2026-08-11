"""Cottrell pinning of a 1/2<111> SCREW dislocation in bcc Ta, for comparison
with the edge results of depin3d.py.

Two things change relative to the edge:
  1. the interaction field: cos(3 theta)/r from anisotropic elasticity
     (exactly zero in the isotropic approximation), ~10x weaker in amplitude
  2. the line tension: Gamma_screw = mu b^2 (1+nu) ln(R/r0) / [4 pi (1-nu)]
     which is 4.2x STIFFER than the edge value (1-2nu) -> harder to pin
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import depin3d as D, stroh_field as SF, sensitivity as S
from depin3d import P, GPA, KB

C11, C12, C44 = 266.0, 158.2, 87.4
K = (C11 + 2*C12)/3.0
Om_rel = 11.45
Pdip = K*GPA*Om_rel
mu_V = (C11-C12+3*C44)/5.0
nu_V = (3*K-2*mu_V)/(2*(3*K+mu_V))

def gamma_screw(p=P):
    return p.mu*p.b**2*(1.0+p.nu)*p.lnRr0/(4.0*np.pi*(1.0-p.nu))

def screw_frame():
    e1 = np.array([1.,-1.,0.])/np.sqrt(2)
    e3 = np.array([1., 1.,1.])/np.sqrt(3)
    e2 = np.cross(e3, e1); e2 /= np.linalg.norm(e2)
    return np.vstack([e1, e2, e3])

def screw_table(U_cap, p=P, h=0.20, dx_max=210.0, dy_max=70.0):
    st = SF.Stroh(SF.rotate_C(SF.cubic_C(C11*GPA, C12*GPA, C44*GPA),
                              screw_frame()))
    gx = np.arange(-dx_max, dx_max+1e-9, h); gy = np.arange(-dy_max, dy_max+1e-9, h)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    U = SF.interaction_field(X, Y, Pdip, st, np.array([0.,0.,p.b]), U_cap)
    return S.FieldTable(U, gx[0], gy[0], h), st

def run(theta, U_cap, seed, Gamma, p=P, tau_max=8.0):
    tab, _ = screw_table(U_cap, p)
    ix = (np.arange(p.Nx)-p.Nx//2)*p.a; iy = (np.arange(p.Ny)-p.Ny//2)*p.a
    X, Y = np.meshgrid(ix, iy, indexing="ij")
    U2d = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U2d[i,j] = S._lookup(tab.U, tab.dx_lo, tab.dy_lo, tab.h, X[i,j], Y[i,j])
    e = np.exp(-U2d/(KB*p.T_age)); ceq = p.c0*e/(1-p.c0+p.c0*e)
    occ = p.c0 + theta*(ceq-p.c0)
    rng = np.random.default_rng(seed); keep = (X**2+Y**2) <= p.R_keep**2
    xs, ys, zs = [], [], []
    for k in range(p.Nz):
        d = (rng.random(occ.shape) < occ) & keep
        gx_, gy_ = np.nonzero(d)
        xs.append(X[gx_,gy_]); ys.append(Y[gx_,gy_]); zs.append(np.full(gx_.size,k))
    xs=np.concatenate(xs); ys=np.concatenate(ys); zs=np.concatenate(zs).astype(np.int64)
    xgrid = np.arange(p.x_lo, p.x_hi+1e-9, p.dx_grid)
    V = S._accumulate_V_tab(xs, ys, zs, xgrid, p.Nz, tab.U, tab.dx_lo, tab.dy_lo, tab.h)
    Fpin = np.ascontiguousarray(-np.gradient(V, p.dx_grid, axis=1))
    Vb = V.mean(axis=0); x0 = float(xgrid[np.argmin(Vb)])
    kG = Gamma/p.a
    x = np.full(p.Nz, x0)
    taus = np.linspace(0.0, tau_max*GPA, p.n_tau)
    for tau in taus:
        if D.relax(x, Fpin, xgrid[0], p.dx_grid, kG, tau*p.b*p.a, p.Bdrag,
                   p.dt, p.n_relax, p.tol_F, p.d_runaway, p.v_stall, p.chunk) == 1:
            return tau/GPA, occ, U2d
    return np.nan, occ, U2d

Ge, Gs = D.gamma_edge(), gamma_screw()
print(f"Gamma_edge  = {Ge:.3f} eV/A")
print(f"Gamma_screw = {Gs:.3f} eV/A   ratio = {Gs/Ge:.2f}x stiffer\n")
print(f"{'theta':>6} {'tau_c screw':>14} {'tau_c edge':>11} {'ratio':>8}")
print("-"*44)
edge = {0.0: 0.155, 0.25: 1.248, 1.0: 5.184}   # depin3d.py, smooth core, 6 seeds
out = {}
for th in [0.0, 0.25, 1.0]:
    r = [run(th, 0.70, P.seed+313*s+int(100*th), Gs, tau_max=0.05)[0] for s in range(4)]
    out[th] = np.mean(r)
    print(f"{th:6.2f} {np.mean(r):9.3f} +/-{np.std(r):4.3f} {edge[th]:11.3f} "
          f"{np.mean(r)/edge[th]:8.3f}")

_, occ, U2d = run(1.0, 0.70, P.seed, Gs, tau_max=0.05)
fig, ax = plt.subplots(1,3, figsize=(16,4.6))
g = (np.arange(P.Nx)-P.Nx//2)*P.a
im=ax[0].imshow(U2d.T, origin="lower", cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                extent=[g[0],g[-1],g[0],g[-1]])
ax[0].set_title(r"(a) screw $U(x,y)$: $\cos 3\theta/r$ [eV]"); fig.colorbar(im,ax=ax[0],shrink=.85)
im=ax[1].imshow(occ.T, origin="lower", cmap="magma", vmin=0, vmax=1,
                extent=[g[0],g[-1],g[0],g[-1]])
ax[1].set_title(r"(b) screw atmosphere, $\theta_{\rm age}$=1: three lobes")
fig.colorbar(im,ax=ax[1],shrink=.85)
for a in ax[:2]:
    a.set_xlim(-25,25); a.set_ylim(-25,25)
    a.set_xlabel(r"$x$ [$\rm\AA$]"); a.set_ylabel(r"$y$ [$\rm\AA$]")
ths=[0.0,0.25,1.0]
ax[2].plot(ths,[edge[t] for t in ths],"o-",lw=2,label="edge")
ax[2].plot(ths,[out[t] for t in ths],"s-",lw=2,label="screw")
ax[2].axhline(P.mu/30/GPA,color="0.35",ls="-.",label=r"$\mu/30$")
ax[2].set_yscale("log"); ax[2].set_xlabel(r"ageing fraction $\theta$")
ax[2].set_ylabel(r"$\tau_c$ (0 K) [GPa]"); ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)
ax[2].set_title("(c) screw vs edge")
fig.suptitle("Cottrell pinning of the RATE-CONTROLLING screw dislocation in bcc Ta(Lu)",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("screw_depin.png",dpi=140)
print("\nfigure written: screw_depin.png")
