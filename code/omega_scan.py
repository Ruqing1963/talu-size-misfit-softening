"""Sensitivity of tau_c to the relaxation volume Omega_rel (the DFT quantity
that actually sets the strength scale)."""
import numpy as np, depin3d as D, stroh_field as SF, sensitivity as S
from depin3d import P, GPA, KB

def table_for_omega(om, U_cap, p=P, h=0.20, dx_max=210.0, dy_max=70.0):
    C11, C12, C44 = 266.0, 158.2, 87.4
    K = (C11 + 2*C12)/3.0
    Pv = K*GPA*om
    st = SF.Stroh(SF.rotate_C(SF.cubic_C(C11*GPA, C12*GPA, C44*GPA),
                              SF.dislocation_frame()))
    gx = np.arange(-dx_max, dx_max+1e-9, h); gy = np.arange(-dy_max, dy_max+1e-9, h)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    U = SF.interaction_field(X, Y, Pv, st, np.array([p.b,0.,0.]), U_cap)
    return S.FieldTable(U, gx[0], gy[0], h)

def run(om, U_cap, theta, seed, p=P):
    tab = table_for_omega(om, U_cap, p)
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
    F = np.ascontiguousarray(-np.gradient(V, p.dx_grid, axis=1))
    Vb = V.mean(axis=0); x0 = float(xgrid[np.argmin(Vb)])
    return D.tau_c_quasistatic(F, xgrid, x0, p)[0]

oms = [7.0, 9.0, 11.45, 14.0, 17.0]
print("Omega_rel [A^3]   tau_c [GPa]   (theta=0.25, U_cap=0.7 eV, aniso, 2 seeds)")
ys = []
for om in oms:
    t = [run(om, 0.70, 0.25, P.seed+313*s+int(om)) for s in range(4)]
    ys.append(np.mean(t))
    print(f"   {om:6.2f}        {np.mean(t):6.3f} +/- {np.std(t):.3f}")
n = np.polyfit(np.log(oms), np.log(ys), 1)[0]
print(f"\n  tau_c ~ Omega_rel^{n:.2f}")
print(f"  => a 10% error in Omega_rel gives a {100*((1.1**n)-1):.1f}% error in tau_c")
print(f"  (Varvenne-Curtin predicts tau_y0 ~ Delta_v^(4/3) = ^1.33)")
