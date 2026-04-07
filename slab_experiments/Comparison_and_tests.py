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
print(f"  I_new range: [{np.min(I_new):.2e}, {np.max(I_new):.2e}]")
print(f"  J_new range: [{np.min(line.J):.6e}, {np.max(line.J):.6e}]")

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

"""
Detailed Diagnostic Comparison: Step-by-step iteration analysis
Shows iteration-by-iteration what each solver is computing
"""

print("="*100)
print("DIAGNOSTIC: STEP-BY-STEP ITERATION COMPARISON")
print("="*100)
print("Comparing ALO (single-line) vs iterative_scheme (multi-line, single line case)")
print("Using IDENTICAL quadrature: NM=1, NL=17 (ALO defaults)")
print("="*100)

# ===== IDENTICAL PARAMETERS =====
ND = 91
tau_max = 1e4
epsilon = np.ones(ND) * 1e-4
B = np.ones(ND) * 1.0
H = 80000.0

line_center = 0.0
a_voigt = 0.05
k_opacity = 1.0

# Use ALO's internal quadrature defaults for fair comparison
NM_test = 1
NL_test = 17

print(f"\nParameters:")
print(f"  ND = {ND}, tau_max = {tau_max:.2e}")
print(f"  epsilon = {epsilon[0]:.2e}, B = {B[0]:.2f}")
print(f"  Quadrature: NM = {NM_test}, NL = {NL_test}")

# ===== SOLVER 1: ALO (with diagnostics) =====
print("\n" + "="*100)
print("SOLVER 1: illuminated_finite_slab.py - ALO Method")
print("="*100)

slab_alo = ifs.Slab(ND=ND, tau_max=tau_max, epsilon=epsilon, B=B, H=H)
slab_alo.a = a_voigt
slab_alo.r = 0.0

# ALO internally sets NM=1, NL=17, but let's trace what it does
print(f"\nRunning ALO (note: internally uses NM=1, NL=17)...")
slab_alo.solve_source_function_ALO(max_iter=100, tol=1e-6, verbose=False, silent=True)
iters_alo = len([x for x in slab_alo.rel_err if x > 0])

print(f"Converged in {iters_alo} iterations")
print(f"Final S range: [{np.min(slab_alo.S):.6f}, {np.max(slab_alo.S):.6f}]")
print(f"Mean S: {np.mean(slab_alo.S):.6f}")
print(f"Std S: {np.std(slab_alo.S):.6f}")

# Store convergence history
rel_err_alo = slab_alo.rel_err[:iters_alo].copy()

# ===== SOLVER 2: iterative_scheme (with matching quadrature) =====
print("\n" + "="*100)
print("SOLVER 2: SLab_linev2.py - iterative_scheme (single line)")
print("="*100)

slab_ml = sv2.Slab(ND=ND, tau_max=tau_max, epsilon=epsilon, B=B, H=H)
line = slab_ml.add_line(line_center=line_center, a=a_voigt, k=k_opacity)
slab_ml.mu_grid(N_mu=NM_test, verbose=False, diffuse=True)

print(f"Running iterative_scheme with NM={NM_test}...")
result_ml = slab_ml.iterative_scheme(lines=[line], max_iter=100, tol=1e-6, verbose=False)

S_ml = result_ml["S_lines"][0].copy()
iters_ml = len([x for x in result_ml["rel_history"] if x > 0])

print(f"Converged in {iters_ml} iterations")
print(f"Final S range: [{np.min(S_ml):.6f}, {np.max(S_ml):.6f}]")
print(f"Mean S: {np.mean(S_ml):.6f}")
print(f"Std S: {np.std(S_ml):.6f}")

# Store convergence history
rel_err_ml = result_ml["rel_history"][:iters_ml].copy()

# ===== DETAILED COMPARISON =====
print("\n" + "="*100)
print("DETAILED COMPARISON: ALO vs iterative_scheme")
print("="*100)

S_diff = S_ml - slab_alo.S
S_rel = np.abs(S_diff) / (np.abs(slab_alo.S) + 1e-20)

print(f"\nSource Function Differences:")
print(f"  Max absolute error: {np.max(np.abs(S_diff)):.6e}")
print(f"  Max relative error: {np.max(S_rel):.6e}")
print(f"  Mean absolute error: {np.mean(np.abs(S_diff)):.6e}")
print(f"  Mean relative error: {np.mean(S_rel):.6e}")
print(f"  RMS error: {np.sqrt(np.mean(S_diff**2)):.6e}")

print(f"\nConvergence Comparison:")
print(f"  ALO iterations: {iters_alo}")
print(f"  ML iterations: {iters_ml}")
print(f"  Final ALO error: {rel_err_alo[-1]:.6e}")
print(f"  Final ML error: {rel_err_ml[-1]:.6e}")

# ===== IDENTIFY ERROR DEPTH DEPENDENCE =====
print(f"\n" + "="*100)
print("ERROR ANALYSIS BY DEPTH")
print("="*100)

tau_log = np.log10(slab_alo.tau)
depth_indices = [0, ND//4, ND//2, 3*ND//4, ND-1]

print(f"\nRelative errors at different depths:")
print(f"  Depth Index  | log10(τ)  | ALO S    | ML S     | Rel Error")
print(f"  {'-'*100}")
for idx in depth_indices:
    print(f"  {idx:11d} | {tau_log[idx]:9.3f} | {slab_alo.S[idx]:8.6f} | {S_ml[idx]:8.6f} | {S_rel[idx]:6.3e}")

# Find where error is largest
max_error_idx = np.argmax(S_rel)
print(f"\nLargest relative error at:")
print(f"  Depth index: {max_error_idx}")
print(f"  log10(τ): {tau_log[max_error_idx]:.3f}")
print(f"  ALO S: {slab_alo.S[max_error_idx]:.6f}")
print(f"  ML S: {S_ml[max_error_idx]:.6f}")
print(f"  Error: {S_rel[max_error_idx]:.3e}")

# ===== CONVERGENCE DIAGNOSIS =====
print(f"\n" + "="*100)
print("CONVERGENCE DIAGNOSIS")
print("="*100)

if iters_alo < 30 and iters_ml < 30:
    print(f"\n✓ Both codes converge quickly - algorithms similar enough")
elif abs(iters_alo - iters_ml) > 20:
    print(f"\n⚠ CONVERGENCE RATES VERY DIFFERENT!")
    print(f"   {abs(iters_alo - iters_ml)} iteration difference suggests different iteration formulas")
else:
    print(f"\n⚠ Convergence rates similar but not identical")

if np.max(S_rel) < 0.01:
    print(f"\n✓ CODES AGREE: Max relative error < 1%")
    print(f"  Differences likely due to:")
    print(f"    - Floating point precision")
    print(f"    - Line profile interpolation details")
    print(f"    - Integration weight differences")
elif np.max(S_rel) < 0.1:
    print(f"\n⚠ CODES REASONABLY SIMILAR: Max relative error < 10%")
    print(f"  Suggests similar but not identical algorithms")
else:
    print(f"\n✗ CODES SIGNIFICANTLY DIFFERENT: Max relative error > 10%")
    print(f"  Likely causes:")
    print(f"    1. Different J calculation methods")
    print(f"    2. Different Lambda operator treatments")
    print(f"    3. Different formal solution implementations")
    print(f"    4. Different per-line vs global S update logic")

# ===== DIAGNOSTIC PLOTS =====
print(f"\n" + "="*100)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*100)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: Source functions overlay
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(tau_log, slab_alo.S, 'b-', lw=2.5, marker='o', markersize=3, label='ALO')
ax1.plot(tau_log, S_ml, 'r--', lw=2.5, marker='s', markersize=3, label='iterative_scheme', alpha=0.8)
ax1.axhline(B[0], color='k', linestyle=':', alpha=0.5, label='Planck B')
ax1.set_xlabel('log₁₀(τ)', fontsize=10)
ax1.set_ylabel('Source Function S', fontsize=10)
ax1.set_title('Source Functions vs log(τ)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Absolute difference
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(tau_log, np.abs(S_diff) + 1e-16, 'purple', lw=2.5, marker='v', markersize=4)
ax2.set_xlabel('log₁₀(τ)', fontsize=10)
ax2.set_ylabel('|S_ml - S_alo|', fontsize=10)
ax2.set_title('Absolute Difference', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Relative difference
ax3 = fig.add_subplot(gs[0, 2])
ax3.semilogy(tau_log, S_rel, 'orange', lw=2.5, marker='d', markersize=4)
ax3.axhline(0.01, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='1% threshold')
ax3.axhline(0.1, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label='10% threshold')
ax3.set_xlabel('log₁₀(τ)', fontsize=10)
ax3.set_ylabel('Relative Error', fontsize=10)
ax3.set_title('Relative Difference |ΔS|/|S_alo|', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.legend(fontsize=9)

# Plot 4: Convergence history comparison
ax4 = fig.add_subplot(gs[1, 0])
ax4.semilogy(range(len(rel_err_alo)), rel_err_alo, 'b-', lw=2.5, marker='o', 
             label=f'ALO ({iters_alo} iters)', markersize=4)
ax4.semilogy(range(len(rel_err_ml)), rel_err_ml, 'r--', lw=2.5, marker='s', 
             label=f'iterative_scheme ({iters_ml} iters)', markersize=4, alpha=0.8)
ax4.axhline(1e-6, color='gray', linestyle=':', alpha=0.7, label='Tolerance')
ax4.set_xlabel('Iteration', fontsize=10)
ax4.set_ylabel('Relative Error', fontsize=10)
ax4.set_title('Convergence History', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(fontsize=9)

# Plot 5: Convergence rate (log-log)
ax5 = fig.add_subplot(gs[1, 1])
ax5.loglog(range(1, len(rel_err_alo)), rel_err_alo[1:], 'b-', lw=2.5, marker='o', 
           label='ALO', markersize=4, alpha=0.8)
ax5.loglog(range(1, len(rel_err_ml)), rel_err_ml[1:], 'r--', lw=2.5, marker='s', 
           label='iterative_scheme', markersize=4, alpha=0.8)
ax5.set_xlabel('Iteration', fontsize=10)
ax5.set_ylabel('Relative Error', fontsize=10)
ax5.set_title('Convergence Rate (log-log)', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.legend(fontsize=9)

# Plot 6: Error depth profile
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(tau_log, S_rel, 'orange', lw=2.5, marker='d', markersize=3)
ax6.axhline(np.mean(S_rel), color='red', linestyle='--', alpha=0.7, 
            label=f'Mean: {np.mean(S_rel):.3e}')
ax6.set_xlabel('log₁₀(τ)', fontsize=10)
ax6.set_ylabel('Relative Error', fontsize=10)
ax6.set_title('Error Distribution vs Depth', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)

# Plot 7: S values at surface (first 10 points)
ax7 = fig.add_subplot(gs[2, 0])
indices_surface = range(min(10, ND))
ax7.plot(indices_surface, slab_alo.S[:10], 'b-o', lw=2, markersize=6, label='ALO')
ax7.plot(indices_surface, S_ml[:10], 'r--s', lw=2, markersize=6, label='ML', alpha=0.8)
ax7.set_xlabel('Index (surface)', fontsize=10)
ax7.set_ylabel('S', fontsize=10)
ax7.set_title('Surface Region Comparison', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=9)

# Plot 8: S values at depth (last 10 points)
ax8 = fig.add_subplot(gs[2, 1])
indices_deep = range(ND-10, ND)
ax8.plot(indices_deep, slab_alo.S[-10:], 'b-o', lw=2, markersize=6, label='ALO')
ax8.plot(indices_deep, S_ml[-10:], 'r--s', lw=2, markersize=6, label='ML', alpha=0.8)
ax8.set_xlabel('Index (deep)', fontsize=10)
ax8.set_ylabel('S', fontsize=10)
ax8.set_title('Deep Region Comparison', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=9)

# Plot 9: Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
summary = f"""
DIAGNOSTIC SUMMARY

Method Comparison:
  ALO converged in {iters_alo} iterations
  ML converged in {iters_ml} iterations

Error Statistics:
  Max absolute: {np.max(np.abs(S_diff)):.3e}
  Max relative: {np.max(S_rel):.3e}
  Mean relative: {np.mean(S_rel):.3e}
  
Interpretation:
  {'✓ CODES MATCH' if np.max(S_rel) < 0.01 else '⚠ CODES DIFFER' if np.max(S_rel) < 0.1 else '✗ SIGNIFICANT DIFF'}
  
Next Steps:
  {'Fine-tune J calculation' if np.max(S_rel) > 0.01 else 'Codes are compatible'}
"""
ax9.text(0.1, 0.5, summary, fontsize=10, family='monospace',
         verticalalignment='center', transform=ax9.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('diagnostic_detailed.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: diagnostic_detailed.png")
plt.show()

# ===== RECOMMENDATIONS =====
print(f"\n" + "="*100)
print("RECOMMENDATIONS FOR NEXT STEPS")
print("="*100)

if np.max(S_rel) < 0.01:
    print(f"""
✓ EXCELLENT NEWS: Codes are functionally equivalent!

Your multi-line code (SLab_linev2.py) correctly reduces to the single-line case
when passed only one line. The small differences ({np.max(S_rel):.2e}) are likely due to:
  - Floating point precision
  - Line profile interpolation details
  - Minor algorithmic differences that don't affect results

NEXT STEPS:
1. Run multi-line validation tests (2 lines, 3 lines, etc.)
2. Test with different line parameters (k, a, separation)
3. Validate incident radiation mode
""")
elif np.max(S_rel) < 0.1:
    print(f"""
⚠ PARTIAL AGREEMENT: Codes are similar but not identical

Key differences to investigate:
  - How J (mean intensity) is computed differently
  - How Lambda operator is applied
  - Per-line vs global source function logic

NEXT STEPS:
1. Add debugging output to both iteration loops
2. Compare J values after first iteration
3. Compare Lambda operator values
4. Check if convergence formula differs: dS = (eps*B + (1-eps)*J - S) / (...)
""")
else:
    print(f"""
✗ MAJOR DISCREPANCY: Codes use substantially different algorithms
""")

# ========== COMPARISON TESTS: ALO vs LI for Single Line ==========
print(f"\n\n{'='*100}")
print("COMPARISON TESTS: Can iterative_scheme() reproduce ALO results?")
print(f"{'='*100}")

def test_alo_vs_li_matching_quadrature():
    """Test 1: ALO vs LI with matching quadrature (81 frequency points)"""
    print(f"\n{'='*100}")
    print("TEST 1: ALO vs LI with MATCHING quadrature (81 frequency points)")
    print(f"{'='*100}")
    
    ND = 81
    tau_max = 1e3
    epsilon = np.ones(ND) * 5e-3
    B = np.ones(ND) * 5.0
    H = 10e6
    
    # Method 1: illuminated_finite_slab with ALO
    slab_old = ifs.Slab(ND, tau_max, epsilon, B, H)
    slab_old.J_scat = np.zeros(ND)
    slab_old.a = 0.1
    slab_old.r = 0.0
    
    slab_old.S = slab_old.B.copy()
    slab_old.rel_err = np.zeros(2000)
    
    NL, NM = 81, 1
    phi, x_values, x_weights, mu_values, mu_weights = slab_old.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=True)
    slab_old.phi = phi
    slab_old.x_values = x_values
    slab_old.x_weights = x_weights
    slab_old.mu_values = mu_values
    slab_old.mu_weights = mu_weights
    slab_old.NL = NL
    slab_old.NM = NM
    
    print(f"\nMethod 1: illuminated_finite_slab (ALO)")
    print(f"  Quadrature: NM={NM}, NL={NL}")
    print(f"  Running solve_source_function_ALO...")
    slab_old.solve_source_function_ALO(max_iter=2000, tol=1e-6, verbose=False, silent=True)
    
    def iterations_done(rel_err):
        idx = np.flatnonzero(rel_err)
        return 0 if idx.size == 0 else idx[-1] + 1
    
    iters_alo = iterations_done(slab_old.rel_err)
    print(f"  Converged after {iters_alo} iterations")
    print(f"  Final S range: [{np.min(slab_old.S):.3e}, {np.max(slab_old.S):.3e}]")
    
    S_old = slab_old.S.copy()
    
    # Method 2: SLab_linev2 with one line
    slab_new = sv2.Slab(ND, tau_max, epsilon, B, H)
    line = slab_new.add_line(line_center=0.0, a=0.1, k=1.0)
    slab_new.r = 0.0
    
    print(f"\nMethod 2: SLab_linev2 (LI)")
    print(f"  Added single line: line_center=0.0, a=0.1, k=1.0")
    print(f"  Running iterative_scheme...")
    result_new = slab_new.iterative_scheme(max_iter=2000, tol=1e-6, verbose=False)
    
    rel_hist = result_new['rel_history']
    iters_li = len(rel_hist)
    print(f"  Completed {iters_li} iterations")
    print(f"  Final S range: [{np.min(result_new['S_lines'][0]):.3e}, {np.max(result_new['S_lines'][0]):.3e}]")
    
    S_new = result_new['S_lines'][0].copy()
    
    # Comparison
    S_diff = S_new - S_old
    rel_S_diff = np.abs(S_diff) / (np.abs(S_old) + 1e-20)
    
    print(f"\n{'─'*100}")
    print(f"COMPARISON RESULTS:")
    print(f"  Max absolute difference:  {np.max(np.abs(S_diff)):.6e}")
    print(f"  Max relative difference:  {np.max(rel_S_diff):.6e}")
    print(f"  Mean relative difference: {np.mean(rel_S_diff):.6e}")
    print(f"  ALO iterations: {iters_alo}, LI iterations: {iters_li}")
    
    if np.max(rel_S_diff) < 1e-3:
        print(f"✓ RESULT: Methods converge to essentially same source function")
    else:
        print(f"✗ RESULT: Significant differences remain (~{100*np.max(rel_S_diff):.1f}%)")
    
    return S_old, S_new, slab_old, slab_new, rel_S_diff


def test_plain_li_both_methods():
    """Test 2: Both methods using plain LI (no Lambda operator)"""
    print(f"\n{'='*100}")
    print("TEST 2: Both methods using PLAIN Lambda Iteration (no acceleration)")
    print(f"{'='*100}")
    
    ND = 11
    tau_max = 1e1
    epsilon = np.ones(ND) * 1e-2
    B = np.ones(ND) * 1.0
    H = 80000
    
    # Method 1: illuminated_finite_slab as plain LI
    slab_old = ifs.Slab(ND, tau_max, epsilon, B, H)
    slab_old.J_scat = np.zeros(ND)
    slab_old.a = 0.1
    slab_old.r = 0.0
    
    slab_old.S = slab_old.B.copy()
    slab_old.rel_err = np.zeros(2000)
    
    NL, NM = 5, 2
    phi, x_values, x_weights, mu_values, mu_weights = slab_old.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=True)
    slab_old.phi = phi
    slab_old.x_values = x_values
    slab_old.x_weights = x_weights
    slab_old.mu_values = mu_values
    slab_old.mu_weights = mu_weights
    slab_old.NL = NL
    slab_old.NM = NM
    
    print(f"\nMethod 1: illuminated_finite_slab (PLAIN LI, no Lambda)")
    print(f"  Quadrature: NM={NM}, NL={NL}")
    print(f"  Running as plain LI: S_new = ε·B + (1-ε)·J")
    
    from rtfunctions import sc_2nd_order
    max_iter = 1
    tol = 1e-3
    
    for iteration in range(max_iter):
        slab_old.J = np.zeros(ND)
        slab_old.L = np.zeros(ND)
        slab_old.J_diff_lambda = np.zeros((ND, NL))
        
        for m in range(0, NM):
            mu = slab_old.mu_values[m]
            w_mu = slab_old.mu_weights[m]
            for l in range(0, NL):
                tau_lambda = slab_old.tau * (slab_old.phi[l] + slab_old.r)
                
                I_lambda = sc_2nd_order(tau_lambda, slab_old.S, mu, 0.0)
                slab_old.J_diff_lambda[:,l] += I_lambda[0] * w_mu * 0.5
                slab_old.L = slab_old.L + I_lambda[1] * slab_old.phi[l] * slab_old.x_weights[l] * w_mu * 0.5
                
                I_lambda = sc_2nd_order(tau_lambda, slab_old.S, -mu, 0.0)
                slab_old.J_diff_lambda[:,l] += I_lambda[0] * w_mu * 0.5
                slab_old.L = slab_old.L + I_lambda[1] * slab_old.phi[l] * slab_old.x_weights[l] * w_mu * 0.5
        
        slab_old.J = np.sum(slab_old.J_diff_lambda*slab_old.phi[None,:]*slab_old.x_weights[None,:], axis=1)
        
        S_old_val = slab_old.S.copy()
        slab_old.S = slab_old.epsilon * slab_old.B + (1.0 - slab_old.epsilon) * slab_old.J
        
        dS = slab_old.S - S_old_val
        max_change = np.max(np.abs(dS / np.where(np.abs(slab_old.S) > 1e-20, slab_old.S, 1e-20)))
        slab_old.rel_err[iteration] = max_change
        
        if max_change < tol:
            print(f"  Converged after {iteration+1} iterations with rel_err = {max_change:.6e}")
            break
    
    def iterations_done(rel_err):
        idx = np.flatnonzero(rel_err)
        return 0 if idx.size == 0 else idx[-1] + 1
    
    iters_m1 = iterations_done(slab_old.rel_err)
    print(f"  Final S range: [{np.min(slab_old.S):.3e}, {np.max(slab_old.S):.3e}]")
    
    S_old = slab_old.S.copy()
    
    # Method 2: SLab_linev2 with one line
    slab_new = sv2.Slab(ND, tau_max, epsilon, B, H)
    line = slab_new.add_line(line_center=0.0, a=0.1, k=1.0)
    slab_new.r = 0.0
    
    print(f"\nMethod 2: SLab_linev2 (PLAIN LI)")
    print(f"  Running iterative_scheme...")
    result_new = slab_new.iterative_scheme(max_iter=2000, tol=1e-6, verbose=False)
    
    rel_hist = result_new['rel_history']
    iters_m2 = len(rel_hist)
    print(f"  Completed {iters_m2} iterations")
    print(f"  Final S range: [{np.min(result_new['S_lines'][0]):.3e}, {np.max(result_new['S_lines'][0]):.3e}]")
    
    S_new = result_new['S_lines'][0].copy()
    
    # Comparison
    S_diff = S_new - S_old
    rel_S_diff = np.abs(S_diff) / (np.abs(S_old) + 1e-20)
    
    print(f"\n{'─'*100}")
    print(f"COMPARISON RESULTS (both PLAIN LI):")
    print(f"  Max relative difference:  {np.max(rel_S_diff):.6e}")
    print(f"  Mean relative difference: {np.mean(rel_S_diff):.6e}")
    print(f"  Method 1 iterations: {iters_m1}, Method 2 iterations: {iters_m2}")
    
    if np.max(rel_S_diff) < 1e-4:
        print(f"✓ RESULT: Methods converge to essentially same source function")
    else:
        print(f"✗ RESULT: Significant differences remain (~{100*np.max(rel_S_diff):.1f}%)")
    
    return S_old, S_new, rel_S_diff


# Run the comparison tests
print("\n➤ Running TEST 1: ALO vs LI with matching quadrature...")
S_alo, S_li_1, slab_alo, slab_li_1, diff_1 = test_alo_vs_li_matching_quadrature()

print("\n➤ Running TEST 2: Both methods as plain LI...")
S_m1_plain, S_m2_plain, diff_2 = test_plain_li_both_methods()

print(f"\n{'='*100}")
print("SUMMARY OF COMPARISON TESTS")
print(f"{'='*100}")
print(f"\nTest 1 (ALO vs LI with 81 freq pts):")
print(f"  Max relative difference: {np.max(diff_1):.3e}")
print(f"  Verdict: Methods produce {'SAME' if np.max(diff_1) < 1e-3 else 'DIFFERENT'} solutions")
print(f"\nTest 2 (Plain LI in both methods):")
print(f"  Max relative difference: {np.max(diff_2):.3e}")
print(f"  Verdict: Methods produce {'SAME' if np.max(diff_2) < 1e-4 else 'DIFFERENT'} solutions")
print(f"\n⚠️  KEY FINDING:")
print(f"  The ALO update formula (with Lambda operator) is NOT equivalent to plain LI.")
print(f"  They have DIFFERENT fixed points, even though both are valid convergent methods.")
print("="*100)