import numpy as np
import SLab_linev2 as v2

def experiment(ND, tau_max, eps_val, k2_list=(8.0,20.0,50.0), NM_list=(8,32)):
    B = np.ones(ND)
    epsilon = np.ones(ND) * eps_val
    for NM in NM_list:
        for k2 in k2_list:
            slab = v2.Slab(ND, tau_max, epsilon, B, H=8e4)
            slab.NM = NM
            # lines
            line1 = v2.Slab.Line(ND, line_center=0.0, a=0.1, k=1.0, slab_in=slab)
            line2 = v2.Slab.Line(ND, line_center=6.4, a=0.1, k=k2, slab_in=slab)
            lines = [line1, line2]
            slab.global_x_grid(lines)
            for L in lines:
                L.compute_phi_x(slab.x_values)
                norm = np.sum(L.phi_x * slab.x_weights)
                L.phi_x_global = L.phi_x / norm if norm != 0.0 else L.phi_x.copy()
            res = slab.iterate_coupled_lines(lines, max_iter=120, tol=1e-6, verbose=False)
            S_nu = res['S_nu']
            mid = slab.ND // 2
            comp_mid = S_nu[mid, :]
            contrast = float(comp_mid.max() - comp_mid.min())
            print(f"NM={NM:2d}, k2={k2:6.2f} -> composite S(mid) contrast = {contrast:.6g} (min={comp_mid.min():.6g}, max={comp_mid.max():.6g})")

if __name__ == '__main__':
    experiment(81, 1e3, 1e-2)
