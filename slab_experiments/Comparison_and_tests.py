import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

# Import both slab implementations
import illuminated_finite_slab as ifs
import SLab_linev2 as sv2

print("="*100)
print("SINGLE-LINE CODE COMPARISON: illuminated_finite_slab.py vs SLab_linev2.py")
print("="*100)
print("Case: Basic NLTE without incident radiation (no J_scat)")
print("="*100)

# ===== IDENTICAL PARAMETERS FOR BOTH CODES =====
print("\nSetup: Identical Parameters for Both Solvers")
print("-"*100)

# Slab parameters (MUST BE IDENTICAL)
ND = 11                          # Number of depth points
tau_max = 1e1                    # Maximum optical depth
epsilon = np.ones(ND) * 1e-4     # Thermalization parameter (uniform)
B = np.ones(ND) * 1.0            # Planck function (uniform)
H = 80000.0                      # Height above surface (km)

# Line parameters (MUST BE IDENTICAL)
line_center = 0.0                # Line center wavelength (Doppler units)
a_voigt = 0.05                   # Voigt damping parameter
k_opacity = 1.0                  # Opacity coefficient

# Quadrature parameters (SHOULD MATCH for valid comparison)
NM_angles = 2                    # Number of mu (angle) points
NL_freqs = 5                    # Number of frequency points
max_iterations = 1            # Maximum iterations
tolerance = 1e-3                 # Convergence tolerance

print(f"Slab parameters:")
print(f"  ND = {ND}, tau_max = {tau_max:.2e}, epsilon = {epsilon[0]:.2e}, B = {B[0]:.2f}")
print(f"  H = {H:.1f} km")
print(f"\nLine parameters:")
print(f"  line_center = {line_center:.2f}, a = {a_voigt:.2f}, k = {k_opacity:.2f}")
print(f"\nQuadrature: NM = {NM_angles}, NL = {NL_freqs}, max_iter = {max_iterations}, tol = {tolerance:.2e}")

# ===== SOLVER 1: illuminated_finite_slab.py (Reference) =====
print("\n" + "="*100)
print("SOLVER 1: illuminated_finite_slab.py (Reference Single-Line Code)")
print("="*100)

slab_ref = ifs.Slab(ND=ND, tau_max=tau_max, epsilon=epsilon, B=B, H=H)
slab_ref.a = a_voigt
slab_ref.r = 0.0

print("\nSetting up quadrature (without J_scat)...")
phi, x_values, x_weights, mu_values, mu_weights = slab_ref.calculate_profiles_and_weights(
    NM_angles, NL_freqs, verbose=False, diffuse=True
)
slab_ref.phi = phi
slab_ref.x_values = x_values
slab_ref.x_weights = x_weights
slab_ref.mu_values = mu_values
slab_ref.mu_weights = mu_weights
slab_ref.NL = NL_freqs
slab_ref.NM = NM_angles

print(f"  x_values: {len(slab_ref.x_values)} frequency points")
print(f"  mu_values: {len(slab_ref.mu_values)} angle points")

# NOTE: NOT calling calculate_J_scat() - basic case without illumination
print(f"Solving source function (ALO, no incident radiation)...")
slab_ref.solve_source_function_ALO(max_iter=max_iterations, tol=tolerance, verbose=False, silent=True)
iters_ref = len([x for x in slab_ref.rel_err if x > 0])
print(f"✓ Converged in {iters_ref} iterations")
print(f"  S_ref range: [{np.min(slab_ref.S):.6f}, {np.max(slab_ref.S):.6f}]")
print(f"  J_ref range: [{np.min(slab_ref.J):.6e}, {np.max(slab_ref.J):.6e}]")

# Compute emergent spectrum
x_obs = np.linspace(-6, 6, 501)
I_ref = slab_ref.formal_solution_given_direction(mu_obs=1.0, x_obs=x_obs, boundary_condition=0.0, recalc_profile=True)
print(f"  I_ref range: [{np.min(I_ref):.2e}, {np.max(I_ref):.2e}]")

results_ref = {
    "tau": slab_ref.tau.copy(),
    "S": slab_ref.S.copy(),
    "x_obs": x_obs.copy(),
    "I": I_ref.copy(),
    "iterations": iters_ref,
}

# ===== SOLVER 2: SLab_linev2.py (New Multi-Line Code) =====
print("\n" + "="*100)
print("SOLVER 2: SLab_linev2.py (New Multi-Line Code, Single Line Case)")
print("="*100)

slab_new = sv2.Slab(ND=ND, tau_max=tau_max, epsilon=epsilon, B=B, H=H)
line = slab_new.add_line(line_center=line_center, a=a_voigt, k=k_opacity)
#line = sv2.Slab.Line(NL_freqs, line_center=line_center, a=a_voigt, k=k_opacity, slab_in=slab_new)
#slab_new.global_x_grid([line])

print(f"Setting up quadrature...")
slab_new.mu_grid(N_mu=NM_angles, verbose=False, diffuse=True)
print(f"  mu_values: {len(slab_new.mu_values)} angle points")

print(f"Running iterative scheme (no incident radiation)...")
result_new = slab_new.iterative_scheme(lines=[line], max_iter=max_iterations, tol=tolerance, verbose=False)

S_new = result_new["S_lines"][0].copy()
I_new = result_new["I_emergent"].copy()
rel_hist_new = result_new["rel_history"]
iters_new = len([x for x in rel_hist_new if x > 0])

print(f"✓ Converged in {iters_new} iterations")
print(f"  S_new range: [{np.min(S_new):.6f}, {np.max(S_new):.6f}]")
print(f"  J_new range: [{np.min(line.J):.6e}, {np.max(line.J):.6e}]")
print(f"  I_new range: [{np.min(I_new):.2e}, {np.max(I_new):.2e}]")

results_new = {
    "tau": slab_new.tau.copy(),
    "S": S_new,
    "x_obs": slab_new.x_values.copy(),
    "I": I_new,
    "iterations": iters_new,
}

# ===== COMPARISON ANALYSIS =====
print("\n" + "="*100)
print("COMPARISON ANALYSIS")
print("="*100)

# Source function comparison
S_diff = results_new["S"] - results_ref["S"]
S_rel_diff = np.abs(S_diff) / (np.abs(results_ref["S"]) + 1e-20)

print(f"\n1. SOURCE FUNCTION COMPARISON:")
print(f"   Max absolute difference: {np.max(np.abs(S_diff)):.2e}")
print(f"   Max relative difference: {np.max(S_rel_diff):.2e}")
print(f"   RMS difference: {np.sqrt(np.mean(S_diff**2)):.2e}")

# Intensity comparison (interpolate to common grid)
f_ref = interp1d(results_ref["x_obs"], results_ref["I"], kind='cubic', bounds_error=False, fill_value='extrapolate')
f_new = interp1d(results_new["x_obs"], results_new["I"], kind='cubic', bounds_error=False, fill_value='extrapolate')

x_common = np.linspace(max(results_ref["x_obs"].min(), results_new["x_obs"].min()),
                        min(results_ref["x_obs"].max(), results_new["x_obs"].max()), 300)
I_ref_common = f_ref(x_common)
I_new_common = f_new(x_common)
I_diff = I_new_common - I_ref_common
I_rel_diff = np.abs(I_diff) / (np.abs(I_ref_common) + 1e-20)

print(f"\n2. EMERGENT INTENSITY COMPARISON:")
print(f"   Max absolute difference: {np.max(np.abs(I_diff)):.2e}")
print(f"   Max relative difference: {np.max(I_rel_diff):.2e}")
print(f"   RMS difference: {np.sqrt(np.mean(I_diff**2)):.2e}")

print(f"\n3. CONVERGENCE HISTORY:")
print(f"   Reference code iterations: {results_ref['iterations']}")
print(f"   New code iterations: {results_new['iterations']}")

# ===== DIAGNOSTIC PLOTS (Source Function vs log(tau)) =====
print(f"\n" + "="*100)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*100)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

tau_log = np.log10(results_ref["tau"])
tau_log_new = np.log10(results_new["tau"])

# Plot 1: Source function vs log(tau)
ax = axes[0, 0]
ax.plot(tau_log, results_ref["S"], 'b-', lw=2.5, marker='o', markersize=4, label='Reference (ifs)')
ax.plot(tau_log_new, results_new["S"], 'r--', lw=2.5, marker='s', markersize=4, label='New (SLab_linev2)')
ax.axhline(B[0], color='k', linestyle=':', alpha=0.5, label='Planck B')
ax.set_xlabel('log₁₀(Optical Depth τ)')
ax.set_ylabel('Source Function S')
ax.set_title('Source Function vs log(τ)')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

# Plot 2: Absolute S difference vs log(tau)
ax = axes[0, 1]
ax.semilogy(tau_log, np.abs(S_diff) + 1e-20, 'purple', lw=2.5, marker='v', markersize=4, label='|S_new - S_ref|')
ax.set_xlabel('log₁₀(Optical Depth τ)')
ax.set_ylabel('Absolute Difference')
ax.set_title('Source Function: Absolute Error vs log(τ)')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)

# Plot 3: Relative S difference vs log(tau)
ax = axes[0, 2]
ax.semilogy(tau_log, S_rel_diff, 'orange', lw=2.5, marker='d', markersize=4, label='Relative Error')
ax.axhline(1e-3, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='0.1% threshold')
ax.axhline(1e-2, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label='1% threshold')
ax.set_xlabel('log₁₀(Optical Depth τ)')
ax.set_ylabel('Relative Error |ΔS|/|S_ref|')
ax.set_title('Source Function: Relative Error vs log(τ)')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)

# Plot 4: Emergent intensity comparison
ax = axes[1, 0]
ax.plot(results_ref["x_obs"], results_ref["I"], 'b-', lw=2.5, label='Reference (ifs)', alpha=0.8)
ax.plot(results_new["x_obs"], results_new["I"], 'r--', lw=2.5, label='New (SLab_linev2)', alpha=0.8)
ax.set_xlabel('x (Doppler units)')
ax.set_ylabel('Emergent Intensity')
ax.set_title('Emergent Intensity Spectrum')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

# Plot 5: Absolute intensity difference
ax = axes[1, 1]
ax.semilogy(x_common, np.abs(I_diff) + 1e-20, 'purple', lw=2.5, marker='v', markersize=3, label='|I_new - I_ref|')
ax.set_xlabel('x (Doppler units)')
ax.set_ylabel('Absolute Difference')
ax.set_title('Emergent Intensity: Absolute Error')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)

# Plot 6: Relative intensity difference
ax = axes[1, 2]
ax.semilogy(x_common, I_rel_diff, 'orange', lw=2.5, marker='d', markersize=3, label='Relative Error')
ax.axhline(1e-3, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='0.1% threshold')
ax.axhline(1e-2, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label='1% threshold')
ax.set_xlabel('x (Doppler units)')
ax.set_ylabel('Relative Error |ΔI|/|I_ref|')
ax.set_title('Emergent Intensity: Relative Error')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('comparison_single_line.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: comparison_single_line.png")
plt.show()

# ===== FINAL VERDICT =====
print("\n" + "="*100)
print("COMPARISON VERDICT")
print("="*100)

max_rel_S = np.max(S_rel_diff)
max_rel_I = np.max(I_rel_diff)

if max_rel_S < 1e-2 and max_rel_I < 1e-2:
    print(f"\n✓ CODES MATCH EXCELLENTLY")
    print(f"  Maximum relative errors:")
    print(f"    Source function: {max_rel_S:.2e} (< 1%)")
    print(f"    Emergent intensity: {max_rel_I:.2e} (< 1%)")
    print(f"\n  ✓ VALIDATION PASSED: SLab_linev2.py correctly reproduces")
    print(f"    illuminated_finite_slab.py for the single-line case")
    verdict = "PASSED ✓"
elif max_rel_S < 0.1 and max_rel_I < 0.1:
    print(f"\n⚠ CODES AGREE REASONABLY (< 10%)")
    print(f"  Maximum relative errors:")
    print(f"    Source function: {max_rel_S:.2e}")
    print(f"    Emergent intensity: {max_rel_I:.2e}")
    print(f"\n  Consider investigating numerical differences in:")
    print(f"    - Quadrature setup (mu, frequency grids)")
    print(f"    - Convergence criteria")
    print(f"    - Iteration method implementations")
    verdict = "WARNING ⚠"
else:
    print(f"\n✗ CODES DISAGREE SIGNIFICANTLY (> 10%)")
    print(f"  Maximum relative errors:")
    print(f"    Source function: {max_rel_S:.2e}")
    print(f"    Emergent intensity: {max_rel_I:.2e}")
    print(f"\n  Priority issues to investigate:")
    print(f"    1. Quadrature setup (mu, frequency grids)")
    print(f"    2. Line profile calculations")
    print(f"    3. Formal solution implementation")
    print(f"    4. Convergence criteria or iteration method")
    verdict = "FAILED ✗"

print("\n" + "="*100)
print(f"Test Status: {verdict}")
print("="*100)
print("\nReady for multi-line validation tests...")
print("="*100)

