import numpy as np
import matplotlib.pyplot as plt
import SLab_linev2 as v2

# Parameters (user-specified default)
ND = 81
tau_max = 1e3
eps_val = 1e-2
k1 = 1.0
k2 = 8.0
line_centers = (0.0, 6.4)

# Build slab and lines
epsilon = np.ones(ND) * eps_val
B = np.ones(ND) * 1.0
slab = v2.Slab(ND, tau_max, epsilon, B, H=8e4)
slab.NM = 8  # angular quadrature used in iteration

line1 = v2.Slab.Line(ND, line_center=line_centers[0], a=0.1, k=k1, slab_in=slab)
line2 = v2.Slab.Line(ND, line_center=line_centers[1], a=0.1, k=k2, slab_in=slab)
lines = [line1, line2]

# Build global grid and ensure phi normalized
slab.global_x_grid(lines)
for L in lines:
    L.compute_phi_x(slab.x_values)
    norm = np.sum(L.phi_x * slab.x_weights)
    L.phi_x_global = L.phi_x / norm if norm != 0.0 else L.phi_x.copy()

# Run coupled iteration
res = slab.iterate_coupled_lines(lines, max_iter=200, tol=1e-6, verbose=False)
S_nu = res['S_nu']
I_emergent = res['I_emergent']
S_lines = res['S_lines']

# Prepare per-frequency diagnostics
x = slab.x_values
ND = slab.ND
mid = ND // 2

# denom(x)
denom = np.zeros_like(x)
for L in lines:
    denom += L.k * L.phi_x_global

# per-line opacity contribution k*phi
kphi = [L.k * L.phi_x_global for L in lines]

# numerator components at mid depth: k*phi(x) * S_line[mid]
num_components_mid = []
for i, L in enumerate(lines):
    Sline = np.asarray(S_lines[i])
    num_components_mid.append((L.k * L.phi_x_global) * Sline[mid])
num_components_mid = np.array(num_components_mid)  # shape (nlines, nfreq)

numerator_mid = np.sum(num_components_mid, axis=0)
composite_mid = S_nu[mid, :]

# fractional contribution per line at mid depth
# avoid divide by zero
denom_num = numerator_mid.copy()
denom_safe = np.where(denom_num == 0.0, 1.0, denom_num)
frac_mid = num_components_mid / denom_safe[None, :]

# Save CSV with key arrays
out_csv = 'per_freq_diag_case_tau{:.0f}_eps{:.0g}_k2{:.0f}.npz'.format(tau_max, eps_val, k2)
np.savez(out_csv, x=x, denom=denom, kphi=kphi, numerator_mid=numerator_mid, composite_mid=composite_mid, frac_mid=frac_mid)
print('Saved diagnostic arrays ->', out_csv)

# Plot 1: denom and k*phi curves
plt.figure(figsize=(8,4))
plt.plot(x, denom, label='denom = sum k phi')
for i, kp in enumerate(kphi):
    plt.plot(x, kp, '--', label=f'k{ i+1 } * phi')
plt.xlabel('x (Doppler units)')
plt.ylabel('Opacity weights')
plt.title('Denominator and per-line k*phi contributions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot1 = f'denom_kphi_tau{int(tau_max)}_eps{eps_val}_k2{int(k2)}.png'
plt.savefig(plot1, dpi=150)
plt.close()
print('Saved', plot1)

# Plot 2: numerator components at mid depth and numerator sum; overlay composite S(mid) scaled
plt.figure(figsize=(8,4))
for i in range(num_components_mid.shape[0]):
    plt.plot(x, num_components_mid[i, :], alpha=0.8, label=f'num comp line {i+1} (mid)')
plt.plot(x, numerator_mid, 'k-', lw=2, label='numerator sum (mid)')
# overlay composite mid on secondary axis scaled to numerator amplitude for visual comparison
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(x, composite_mid, 'k--', lw=1.2, label='composite S (mid)', color='tab:orange')
ax2.set_ylabel('Composite S (mid)')
ax1.set_xlabel('x (Doppler units)')
ax1.set_ylabel('k*phi * S_line(mid) (arbitrary units)')
plt.title('Numerator components (mid depth) and Composite S (mid)')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.grid(True)
plt.tight_layout()
plot2 = f'num_components_and_Smid_tau{int(tau_max)}_eps{eps_val}_k2{int(k2)}.png'
plt.savefig(plot2, dpi=150)
plt.close()
print('Saved', plot2)

# Plot 3: fractional contributions (stackplot) at mid depth and overlay composite S(mid) scaled
plt.figure(figsize=(10,4))
labels = [f'Line {i+1}' for i in range(frac_mid.shape[0])]
colors = ['#00bcd4', '#8bc34a', '#ff7043', '#9c27b0']
plt.stackplot(x, frac_mid, labels=labels, colors=colors[:frac_mid.shape[0]], alpha=0.9)
# overlay composite S(mid) scaled to [0,1] range to compare peaks
comp_scaled = (composite_mid - composite_mid.min()) / (composite_mid.max() - composite_mid.min() + 1e-30)
plt.plot(x, comp_scaled, '--k', lw=2, label='composite S(mid) scaled')
plt.xlabel('x (Doppler units)')
plt.ylabel('Fractional contribution (mid depth)')
plt.title('Per-line fractional contributions at mid depth (stacked)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plot3 = f'frac_stack_mid_tau{int(tau_max)}_eps{eps_val}_k2{int(k2)}.png'
plt.savefig(plot3, dpi=150)
plt.close()
print('Saved', plot3)

print('Diagnostic plots written: ', plot1, plot2, plot3)
