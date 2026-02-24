#!/usr/bin/env python3
"""
Sweep script: create slabs for combinations of tau_max and epsilon, run iterate_coupled_lines
with two spectral lines (center=0.0,k=1.0 and center=6.4,k=8.0), and plot results.

Usage:
    python sweep_tau_epsilon.py

Outputs (saved to ./sweep_results/):
 - emergent_spectra_overlay.png : overlayed emergent spectra for every combination
 - S_nu_heatmap_tau{tau}_eps{eps}.png : composite S_nu heatmap per combination
 - results.npz (optional) : numpy archive with emergent spectra and S_nu arrays

This script uses SLab_linev2.Slab and iterate_coupled_lines. It does NOT calculate J_scat by default.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import local module
import SLab_linev2 as v2
import csv

# Output directory
OUTDIR = os.path.join(os.path.dirname(__file__), 'sweep_results')
os.makedirs(OUTDIR, exist_ok=True)

# Sweep parameters
ND = 81  # depth grid points
H = 8e6  # slab height (unused in current solver but required by constructor)
B_val = 1.0

# User-specified lists (feel free to change)
TAU_LIST = [1e1, 1e2, 1e3, 1e4]
EPS_LIST = [1e-4, 1e-1, 1e-2, 1e-4]

# Line definitions (fixed for the sweep)
LINE_PARAMS = [
    dict(line_center=0.0, a=0.15, k=1.0),
    dict(line_center=8.4, a=0.25, k=8.0)
]

# Solver settings
MAX_ITER = 1000
TOL = 1e-6
VERBOSE = False

# Containers for saving results
results = []
summary_rows = []
k_sweep_rows = []

# Parameters for automated checks and targeted k-sweep
STRUCTURE_CONTRAST_THRESHOLD = 0.02  # relative contrast at line center to continuum to call 'structure'
K_SWEEP_FACTORS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
MAX_ITER_SMALL = 300
TOL_SMALL = 1e-4

print('Sweep started:', datetime.utcnow().isoformat(), 'UTC')

for tau_max in TAU_LIST:
    for eps in EPS_LIST:
        print(f"\nBuilding slab: tau_max={tau_max}, epsilon={eps}")

        # create slab
        epsilon_arr = np.ones(ND) * eps
        B = np.ones(ND) * B_val
        slab = v2.Slab(ND, tau_max, epsilon_arr, B, H)

        # create lines
        lines = []
        for p in LINE_PARAMS:
            L = v2.Slab.Line(ND, line_center=p['line_center'], a=p['a'], k=p['k'], slab_in=slab)
            lines.append(L)

        # build global x-grid and compute phi mappings
        slab.global_x_grid(lines)
        for L in lines:
            if hasattr(L, 'compute_phi_x'):
                L.compute_phi_x(slab.x_values)
            # normalize phi on slab.x_weights if available
            if getattr(slab, 'x_weights', None) is None:
                dx = slab.x_values[1] - slab.x_values[0]
                slab.x_weights = np.ones_like(slab.x_values) * dx
            norm = np.sum(L.phi_x * slab.x_weights)
            L.phi_x_global = (L.phi_x / norm) if norm != 0.0 else L.phi_x.copy()
            # Compute S_line
            L.S_line = L.compute_S_line(max_iter=MAX_ITER, tol=TOL, global_x_grid = L.phi_x_global)

        # Optional: do NOT calculate J_scat here; user will test J_scat separately
        # Run coupled iteration
        print(' Running coupled iteration...')
        try:
            res = slab.iterate_coupled_lines(lines, max_iter=MAX_ITER, tol=TOL, verbose=VERBOSE)
        except Exception as e:
            print(' iterate_coupled_lines failed:', e)
            continue

        S_nu = res.get('S_nu')          # shape (ND, Nfreq)
        I_emergent = res.get('I_emergent')
        S_lines = res.get('S_lines')    # list

        x = slab.x_values
        tau = slab.tau

        # Save a heatmap of composite S_nu (depth x frequency)
        fig, ax = plt.subplots(figsize=(8,5))
        extent = [x[0], x[-1], tau[0], tau[-1]]
        im = ax.imshow(S_nu, origin='lower', aspect='auto', extent=extent, cmap='viridis')
        ax.set_xlabel('x (Doppler units)')
        ax.set_ylabel('Optical depth (tau)')
        ax.set_title(f'Composite S_nu: tau_max={tau_max}, eps={eps}')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('S_nu')
        fname = os.path.join(OUTDIR, f'S_nu_heatmap_tau{tau_max}_eps{eps}.png')
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(' Saved S_nu heatmap ->', fname)

        # Save emergent spectrum plot
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(x, I_emergent, lw=2)
        ax.set_xlabel('x (Doppler units)')
        ax.set_ylabel('Emergent intensity (arb)')
        ax.set_title(f'Emergent I: tau_max={tau_max}, eps={eps}')
        # mark line centers
        centers = [p['line_center'] for p in LINE_PARAMS]
        center_idxs = [int(np.argmin(np.abs(x - c))) for c in centers]
        ax.scatter([x[i] for i in center_idxs], [I_emergent[i] for i in center_idxs], color='red', zorder=5)
        fname2 = os.path.join(OUTDIR, f'emergent_tau{tau_max}_eps{eps}.png')
        fig.tight_layout()
        fig.savefig(fname2, dpi=150)
        plt.close(fig)
        print(' Saved emergent spectrum ->', fname2)

        # Collect numeric diagnostics
        denom = np.zeros_like(x)
        for L in lines:
            denom += L.k * L.phi_x_global
        diag = dict(tau_max=float(tau_max), epsilon=float(eps), x=x, tau=tau,
                    I_emergent=I_emergent, S_nu=S_nu, S_lines=S_lines,
                    denom=denom)
        results.append(diag)

        # --- Automated numeric summary for this run ---
        # compute tau_eff (relative) at line centers and max(k*phi) per line
        centers = [L.line_center for L in lines]
        center_idxs = [int(np.argmin(np.abs(x - c))) for c in centers]
        tau_mid = float(tau[slab.ND // 2])
        tau_eff = tau_mid * (denom / (np.max(denom) + 1e-30))
        max_kphi = [float(np.max(L.k * L.phi_x_global)) for L in lines]
        kphi_at_centers = [float(lines[i].k * lines[i].phi_x_global[center_idxs[i]]) for i in range(len(lines))]

        # S_line stats
        Sline_stats = []
        for i, Sline in enumerate(S_lines):
            Sline = np.asarray(Sline)
            Sline_stats.append((float(Sline[0]), float(Sline[slab.ND//2]), float(Sline[-1])))

        # contrast at centers compared to continuum (mid-depth slice)
        comp_mid = S_nu[slab.ND // 2, :]
        contrasts = []
        for i, idx in enumerate(center_idxs):
            # continuum = median excluding ±3 Doppler widths around center
            dx_local = np.abs(x - centers[i])
            cont_mask = dx_local > 3.0
            cont_med = float(np.median(comp_mid[cont_mask])) if np.any(cont_mask) else float(np.median(comp_mid))
            contrast = abs(float(comp_mid[idx]) - cont_med) / (cont_med + 1e-30)
            contrasts.append(contrast)

        structure_detected = any([c > STRUCTURE_CONTRAST_THRESHOLD for c in contrasts])

        summary_row = {
            'tau_max': float(tau_max),
            'epsilon': float(eps),
            'denom_min': float(np.min(denom)),
            'denom_max': float(np.max(denom)),
            'tau_eff_centers': [float(tau_eff[idx]) for idx in center_idxs],
            'max_kphi': max_kphi,
            'kphi_at_centers': kphi_at_centers,
            'Sline_top_mid_bot': Sline_stats,
            'contrasts': contrasts,
            'structure_detected': bool(structure_detected)
        }
        summary_rows.append(summary_row)

        # --- Targeted k-sweep: scale both lines' k by factors until structure appears ---
        found_factor = None
        for fac in K_SWEEP_FACTORS:
            # build new slab and lines but reuse same geometry
            slab_k = v2.Slab(ND, tau_max, epsilon_arr, B, H)
            lines_k = []
            for p in LINE_PARAMS:
                Lk = v2.Slab.Line(ND, line_center=p['line_center'], a=p['a'], k=p['k'] * fac, slab_in=slab_k)
                lines_k.append(Lk)
            slab_k.global_x_grid(lines_k)
            for Lk in lines_k:
                if hasattr(Lk, 'compute_phi_x'):
                    Lk.compute_phi_x(slab_k.x_values)
                if getattr(slab_k, 'x_weights', None) is None:
                    dx = slab_k.x_values[1] - slab_k.x_values[0]
                    slab_k.x_weights = np.ones_like(slab_k.x_values) * dx
                norm = np.sum(Lk.phi_x * slab_k.x_weights)
                Lk.phi_x_global = (Lk.phi_x / norm) if norm != 0.0 else Lk.phi_x.copy()

            try:
                res_k = slab_k.iterate_coupled_lines(lines_k, max_iter=MAX_ITER_SMALL, tol=TOL_SMALL, verbose=False)
            except Exception as _e:
                print(' k-sweep iterate failed for fac', fac, '->', _e)
                continue

            S_nu_k = res_k.get('S_nu')
            comp_mid_k = S_nu_k[slab_k.ND // 2, :]
            # compute contrasts for this k
            contrasts_k = []
            xk = slab_k.x_values
            for i, c in enumerate([Lk.line_center for Lk in lines_k]):
                idxk = int(np.argmin(np.abs(xk - c)))
                dx_local = np.abs(xk - c)
                cont_mask = dx_local > 3.0
                cont_med = float(np.median(comp_mid_k[cont_mask])) if np.any(cont_mask) else float(np.median(comp_mid_k))
                contrast_k = abs(float(comp_mid_k[idxk]) - cont_med) / (cont_med + 1e-30)
                contrasts_k.append(contrast_k)

            row_k = {'tau_max': float(tau_max), 'epsilon': float(eps), 'factor': float(fac), 'contrasts': contrasts_k}
            k_sweep_rows.append(row_k)

            if any([c > STRUCTURE_CONTRAST_THRESHOLD for c in contrasts_k]):
                found_factor = fac
                break

        # record where structure first appeared (or None)
        summary_rows[-1]['structure_detected_at_factor'] = found_factor

        # --- Per-line contribution plots (depth x frequency overlays) ---
        try:
            # build per-line emissivity arrays (depth x freq): eta = k * phi * S_line(depth)
            eta_list = []
            for i, L in enumerate(lines):
                Svec = np.asarray(S_lines[i]).reshape((slab.ND, 1))
                eta = (L.k * L.phi_x_global)[None, :] * Svec  # shape (ND, Nfreq)
                eta_list.append(eta)

            # denom-based contributions (as used in composite S construction)
            denom_safe = np.where(denom == 0.0, 1.0, denom)
            contribs = [eta / denom_safe[None, :] for eta in eta_list]

            # Plot per-line contributions vs x for ALL depths (decimated curves to avoid overplotting)
            fig, ax = plt.subplots(figsize=(12,6))
            palette = ['cyan', 'lime', 'maroon', 'magenta', 'yellow', 'orange']
            decim = max(1, int(np.ceil(slab.ND / 60)))  # aim for ~60 curves
            depths_all = np.arange(0, slab.ND, decim)
            labels_handles = []

            for idx, contrib in enumerate(contribs):
                color = palette[idx % len(palette)]
                # plot many depth slices with low alpha
                for d in depths_all:
                    ax.plot(x, contrib[d, :], color=color, alpha=0.08, linewidth=1)

                # overlay a thicker mean contribution curve for visibility
                mean_curve = np.mean(contrib[depths_all, :], axis=0)
                ax.plot(x, mean_curve, color=color, lw=1.5, alpha=0.9, label=f'Line @{lines[idx].line_center} contribution (avg depths)')

                # find depth where this line contributes the most (integrated over x with weights)
                xw = getattr(slab, 'x_weights', None)
                if xw is None:
                    dx = x[1] - x[0]
                    xw = np.ones_like(x) * dx
                total_per_depth = np.sum(contrib * xw[None, :], axis=1)
                depth_max = int(np.argmax(total_per_depth))
                # plot the contribution curve at that depth as a thick dashed line
                ax.plot(x, contrib[depth_max, :], color=color, lw=2.5, alpha=1.0, linestyle='--',
                        label=f'Line @{lines[idx].line_center} peak depth idx={depth_max}, tau={slab.tau[depth_max]:.2g}')

            # overlay composite S (choose mid depth) as dashed black curve for comparison
            mid = slab.ND // 2
            ax.plot(x, S_nu[mid, :], '--k', lw=2.0, label='Composite S (mid depth)')
            # overlay Planck B reference (mean)
            ax.plot(x, np.mean(B) * np.ones_like(x), ':k', label='Planck B (mean)')

            ax.set_xlabel('x (Doppler units)')
            ax.set_ylabel('Per-line contribution to composite S')
            ax.set_title(f'Per-line contributions vs x (all depths overlay) — tau={tau_max}, eps={eps}')
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True)
            fig.tight_layout()
            fname3 = os.path.join(OUTDIR, f'per_line_contribs_tau{tau_max}_eps{eps}.png')
            fig.savefig(fname3, dpi=150)
            plt.close(fig)
            print(' Saved per-line contribution plot ->', fname3)
        except Exception as e:
            print(' Failed to create per-line contribution plot:', e)

# After sweep, create an overlay plot of emergent spectra for all runs
plt.figure(figsize=(10,6))
for entry in results:
    label = f"tau={entry['tau_max']}, eps={entry['epsilon']}"
    plt.plot(entry['x'], entry['I_emergent'], label=label)
plt.xlabel('x (Doppler units)')
plt.ylabel('Emergent intensity (arb)')
plt.title('Sweep: emergent spectra for tau/epsilon combinations')
plt.legend()
plt.grid(True)
out_overlay = os.path.join(OUTDIR, 'emergent_spectra_overlay.png')
plt.tight_layout()
plt.savefig(out_overlay, dpi=150)
plt.close()
print('\nSaved overlay plot ->', out_overlay)

# Save numeric results to npz
npz_path = os.path.join(OUTDIR, 'sweep_results.npz')
# We can't store variable-length lists directly in npz easily; store as np.savez with compressed arrays per entry
np.savez_compressed(npz_path, results=results)
print('Saved numeric results (npz) ->', npz_path)

# Write summary CSV
csv_path = os.path.join(OUTDIR, 'sweep_summary.csv')
with open(csv_path, 'w', newline='') as cf:
    writer = csv.writer(cf)
    # header
    writer.writerow(['tau_max', 'epsilon', 'denom_min', 'denom_max', 'tau_eff_centers', 'max_kphi', 'kphi_at_centers', 'Sline_top_mid_bot', 'contrasts', 'structure_detected', 'structure_detected_at_factor'])
    for r in summary_rows:
        writer.writerow([r['tau_max'], r['epsilon'], r['denom_min'], r['denom_max'],
                         ';'.join([f"{v:.6g}" for v in r['tau_eff_centers']]),
                         ';'.join([f"{v:.6g}" for v in r['max_kphi']]),
                         ';'.join([f"{v:.6g}" for v in r['kphi_at_centers']]),
                         ';'.join([','.join([f"{vv:.6g}" for vv in triple]) for triple in r['Sline_top_mid_bot']]),
                         ';'.join([f"{v:.6g}" for v in r['contrasts']]),
                         int(r['structure_detected']), r.get('structure_detected_at_factor', '')])
print('Saved sweep summary CSV ->', csv_path)

# Write k-sweep CSV
csv_k_path = os.path.join(OUTDIR, 'k_sweep_summary.csv')
with open(csv_k_path, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['tau_max', 'epsilon', 'factor', 'contrasts'])
    for r in k_sweep_rows:
        writer.writerow([r['tau_max'], r['epsilon'], r['factor'], ';'.join([f"{v:.6g}" for v in r['contrasts']])])
print('Saved k-sweep summary CSV ->', csv_k_path)
print('Sweep finished:', datetime.utcnow().isoformat(), 'UTC')
print('Plots and results are in', OUTDIR)
