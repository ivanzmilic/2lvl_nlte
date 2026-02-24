import numpy as np
import SLab_linev2 as v2

ND = 81
tau = 1000.0
eps = 1e-2
B = np.ones(ND)
slab = v2.Slab(ND, tau, np.ones(ND)*eps, B, H=8e4)
slab.NM = 8
line1 = v2.Slab.Line(ND, line_center=0.0, a=0.1, k=1.0, slab_in=slab)
line2 = v2.Slab.Line(ND, line_center=6.4, a=0.1, k=8.0, slab_in=slab)
# assign per-line epsilon and B (different components)
line1.epsilon = np.ones(ND) * 1e-1
line2.epsilon = np.ones(ND) * 1e-4
line1.B = np.ones(ND) * 0.5
line2.B = np.ones(ND) * 1.5
lines = [line1, line2]
slab.global_x_grid(lines)
for L in lines:
    L.compute_phi_x(slab.x_values)
    norm = np.sum(L.phi_x * slab.x_weights)
    L.phi_x_global = L.phi_x / norm if norm != 0.0 else L.phi_x.copy()
res = slab.iterate_coupled_lines(lines, max_iter=200, tol=1e-6, verbose=False)
S_nu = res['S_nu']
mid = ND//2
comp_mid = S_nu[mid,:]
print(f"Composite S(mid) min,max,contrast = {comp_mid.min():.6g}, {comp_mid.max():.6g}, {float(comp_mid.max()-comp_mid.min()):.6g}")
import matplotlib.pyplot as plt
plt.plot(slab.x_values, comp_mid)
plt.title('Composite S(mid)')
plt.savefig('exp2_composite_mid.png', dpi=150)
print('Saved exp2_composite_mid.png')
