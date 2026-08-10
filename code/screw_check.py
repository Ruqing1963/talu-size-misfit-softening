"""Does a 1/2<111> screw dislocation in bcc Ta bind a substitutional solute?

Tests the two candidate first-order couplings with the Stroh machinery already
validated in stroh_field.py, then estimates the leading second-order term.
"""
import numpy as np, stroh_field as SF
GPA = 6.2415e-3
C11, C12, C44 = 266.0, 158.2, 87.4
a0 = 3.306; b = np.sqrt(3)/2*a0
K = (C11 + 2*C12)/3.0
Om_rel = 11.45
Pdip = K*GPA*Om_rel

def screw_frame():
    e1 = np.array([1.,-1.,0.])/np.sqrt(2)
    e3 = np.array([1., 1.,1.])/np.sqrt(3)      # line AND Burgers vector
    e2 = np.cross(e3, e1); e2 /= np.linalg.norm(e2)
    return np.vstack([e1, e2, e3])

def edge_frame():
    return SF.dislocation_frame()

Cc = SF.cubic_C(C11*GPA, C12*GPA, C44*GPA)
mu_V = (C11-C12+3*C44)/5.0
nu_V = (3*K-2*mu_V)/(2*(3*K+mu_V))
Ci = SF.isotropic_C(mu_V*GPA, nu_V)

th = np.linspace(0.02, 2*np.pi-0.02, 720)
r  = 5.0
x, y = r*np.cos(th), r*np.sin(th)

print(f"probe radius r = {r} A,  b = {b:.3f} A,  P = {Pdip:.2f} eV\n")
print(f"{'case':>34} {'max|eps_kk|':>13} {'max|U| [meV]':>14}")
print("-"*64)
for label, frame, bv, C in [
    ("EDGE  1/2[111](1-10), anisotropic", edge_frame(),  np.array([b,0,0]), Cc),
    ("SCREW 1/2[111],       isotropic",   screw_frame(), np.array([0,0,b]), Ci),
    ("SCREW 1/2[111],       anisotropic", screw_frame(), np.array([0,0,b]), Cc),
]:
    st = SF.Stroh(SF.rotate_C(C, frame))
    ekk = st.dilatation(x, y, bv)
    U = -Pdip*ekk
    print(f"{label:>34} {np.abs(ekk).max():13.3e} {1000*np.abs(U).max():14.4f}")

# --- second-order (diaelastic / modulus) coupling for the screw
mu_Ta, mu_Lu = 69.0, 27.2                 # GPa
Omega_at = 18.07                          # A^3
alpha = abs(mu_Lu-mu_Ta)*GPA*Omega_at     # eV, order-of-magnitude
eps_s = b/(4*np.pi*r)                     # screw shear strain at r
U2 = 0.5*alpha*eps_s**2
print(f"\nsecond-order (modulus) coupling for the screw at r = {r} A:")
print(f"  alpha ~ |mu_Lu - mu_Ta| * Omega = {alpha:.2f} eV")
print(f"  eps_shear = b/(4 pi r) = {eps_s:.4f}")
print(f"  U2 = 1/2 alpha eps^2 = {1000*U2:.2f} meV")
for T in [300, 600]:
    print(f"  vs k_B T at {T} K = {1000*8.617e-5*T:.1f} meV   "
          f"-> U2/kT = {U2/(8.617e-5*T):.2f}")
