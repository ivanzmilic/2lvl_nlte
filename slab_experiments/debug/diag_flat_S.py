import numpy as np
import SLab_linev2 as v2
import numpy as np
import SLab_linev2 as v2


def run_case(ND, tau_max, eps_val, line_centers=(0.0, 6.4), ks=(1.0, 8.0)):
    epsilon = np.ones(ND) * eps_val
    B = np.ones(ND) * 1.0
    slab = v2.Slab(ND, tau_max, epsilon, B, H=8e4)
    lines = []
    for lc, k in zip(line_centers, ks):
        lines.append(v2.Slab.Line(ND, line_center=lc, a=0.1, k=k, slab_in=slab))
    slab.global_x_grid(lines)
    # ensure phi mapping
    for L in lines:
        L.compute_phi_x(slab.x_values)
        norm = np.sum(L.phi_x * slab.x_weights)
        L.phi_x_global = L.phi_x / norm if norm != 0.0 else L.phi_x.copy()
    # run coupled iteration (short)
    res = slab.iterate_coupled_lines(lines, max_iter=80, tol=1e-5, verbose=False)
    S_nu = res['S_nu']
    I_emergent = res['I_emergent']
    S_lines = res['S_lines']
    # diagnostics
    print(f"CASE: ND={ND}, tau_max={tau_max}, eps={eps_val}")
    # denom
    denom = np.zeros_like(slab.x_values)
    for L in lines:
        denom += L.k * L.phi_x_global
    print(f" denom min,max = {np.min(denom):.3e}, {np.max(denom):.3e}")
    # per-line S stats
    for i, Sline in enumerate(S_lines):
        Sline = np.asarray(Sline)
        print(f" Line {i+1} S_line min,max,mean = {Sline.min():.6g}, {Sline.max():.6g}, {Sline.mean():.6g}")
        print(f"  S_line contrast (max-min) = {float(Sline.max()-Sline.min()):.6g}")
    # composite S across x at mid depth
    mid = slab.ND // 2
    comp_mid = S_nu[mid, :]
    print(f" Composite S(mid) min,max,contrast = {comp_mid.min():.6g}, {comp_mid.max():.6g}, {float(comp_mid.max()-comp_mid.min()):.6g}")
    # per-line centers slices
    top = 0
    bot = slab.ND - 1
    for i, L in enumerate(lines):
        idx = int(np.argmin(np.abs(slab.x_values - L.line_center)))
        print(f" Line {i+1} center x={L.line_center} idx={idx}: S_nu top,mid,bot = {S_nu[top, idx]:.6g}, {S_nu[mid, idx]:.6g}, {S_nu[bot, idx]:.6g}")
        print(f"  k*phi(center) = {L.k * L.phi_x_global[idx]:.6g}, denom(center) = {denom[idx]:.6g}")
    # emergent I stats
    print(f" Emergent I min,max = {I_emergent.min():.6g}, {I_emergent.max():.6g}")
    print('-'*60)


if __name__ == '__main__':
    ND = 81
    tau_list = [10.0, 100.0, 1000.0]
    eps_list = [1e-1, 1e-2, 1e-3]
    for tau in tau_list:
        for eps in eps_list:
            run_case(ND, tau, eps)
