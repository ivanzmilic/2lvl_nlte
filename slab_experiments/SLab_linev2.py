from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line
from scipy.stats import norm
import astropy.units as units
import astropy.constants as const
from scipy.special import wofz
from tqdm import tqdm
from rtfunctions import one_full_fs, sc_2nd_order, calc_lambda_full, calc_lambda_monoc
import time
import illuminated_finite_slab as ills

# Define the workflow >>>>>
# 1. First we want to define two classes, namely Slab and Line

# 2. The Slab will have "global" properties, such as global wavelength grid 
# and the Line will have "local" properties, such as the line center, the Doppler width, and the optical depth at the line center.
# 3. The major problem will be how to concatenate the local x grids into the global x grid.

# 4. The base idea is to form the global x grid by contatenating the local x grids, and then sort the global x grid.

# 5. Note that normalization will be a problem, since the local x grids will 
# have different step sizes, and the global x grid will have a step size that is not uniform.

# 6. Line need to access the Slab in order to calculate J (slab's intensity),
# so that it can calculate its own source function.

# 7. The Slab will need to access the Line in order to calculate the emergent spectrum, since the Line will provide the opacity and the source function.

# 8. The tau grid for ALO/LI method will be tau = tau_0 * phi1(x) + tau_0 * k *phi2(x)

# 9. Calulate the J^scat for each line , that is a constant, incoming radiation field which is basically
# the incoming intensity, attenuated by the optical depth in the slab. This is >>constant<<
# for given slab model.


class Slab:
    def __init__(self, ND, tau_max, epsilon, B, H):
        self.ND = ND # number of depth points
        self.tau_max = tau_max # maximum optical depth
        self.epsilon = epsilon # thermalization parameter
        self.B = B # Planck function
        self.H = H # height at which the slab is illuminated
        self.tau_grid = np.linspace(0, tau_max, ND) # optical depth grid
        self.NM = None  # Number of mu points, to be set later when we generate the mu grid
        self.S = np.zeros(ND)  # Source function, to be calculated later
        self.I = np.zeros(ND) # Intensity
        # More parameters:
        self.mu_values = None
        self.mu_weights = None
        self.x_values = None
        self.x_weights = None

        self.compute_tau()  # Compute the tau grid, this function can also be used externally

    class Line:
        def __init__(self, NL, line_center, a, k, r, slab_in):
            self.line_center = line_center
            self.a = a # Damping parameter
            self.k = k # Ratio of the second line to the first line
            self.r = r # Ratio of the line opacity to the continuum opacity at line center
            self.slab_in = slab_in # Reference to the slab instance, so that we can access the slab's properties
            #self.tau_0 = tau_0  # Optical depth at line center
            self.x_grid = None  # Local x grid for this line
            self.phi_x = None  # Line profile function evaluated at x_grid
            self.NL = NL # Number of points in the local x grid for this line

            # Line characteristics
            self.J_scat = None  # Scattered radiation field, to be calculated later
            self.J_diff = None  # Diffuse radiation field, to be calculated later
            self.J = None  # Total radiation field, to be calculated later
            self.S_line = None  # Source function for this line, to be calculated later
            self.norm = None  # Normalization factor for the line profile, to be calculated later
            self.x_weights = None  # Weights for the local x grid, to be calculated later

            self.global_norm = None  # Normalization factor for the line profile on the global x grid, to be calculated later
            self.global_x_idx = None  # Indices of the local x grid points in the global x grid, to be calculated later
            self.global_x_values = None  # Global x values corresponding to the local x grid points
            self.global_x_weights = None  # Global x weights corresponding to the local x grid points
            # Inherited properties from the slab
            self.B = slab_in.B # Planck function
            self.epsilon = slab_in.epsilon # Thermalization parameter, same as the slab's epsilon
        
        def local_x_grid(self, extent=25.0):
            """Generate local x-grid for this line.
            
            Parameters:
            -----------
            extent : float
                Distance from line center to grid boundaries (±extent).
                Default 25.0 to capture full profile without truncation errors.
            """
            start = self.line_center - extent  # Start of the local x grid
            end = self.line_center + extent    # End of the local x grid
            self.x_grid = np.linspace(start, end, self.NL)  # Local x grid for this line

        def compute_phi_x(self, x, type="voigt"):
            """Compute the line profile at wavelength grid points.
            
            CRITICAL: Profile must be computed relative to line center!
            """
            if type == "voigt":
                # Compute Voigt profile relative to line center (CRITICAL FIX)
                x_relative = x - self.line_center
                z = x_relative + 1j * self.a  # Complex argument for the Voigt profile
                self.phi_x = np.real(wofz(z)) / (np.sqrt(np.pi))
            elif type == "gaussian":
                # Compute Gaussian profile relative to line center
                x_relative = x - self.line_center
                self.phi_x = np.exp(-x_relative**2) / np.sqrt(np.pi)  # Gaussian profile
            else:
                raise ValueError("Unknown line profile type")
            
        def compute_weights(self, verbose=False):
            self.compute_phi_x(self.x_grid)  # Compute the line profile at the local x grid
            x_weights = np.ones_like(self.x_grid) * (self.x_grid[-1]-self.x_grid[0]) / len(self.x_grid)
            norm = np.sum(self.phi_x * x_weights)  # Normalization factor for the line profile
            self.norm = norm  # Store the normalization factor
            self.x_weights = x_weights / self.norm  # Store the weights for the local x grid
            if verbose:
                print(f"Weights for local x grid: {self.x_weights}")

        # In case we want J_scat
        def compute_J_scat(self):
             # This function will compute the J_scat for this line
            ND = self.slab_in.ND
            J_inc = np.zeros(ND)
            self.slab_in.mu_grid(self.slab_in.ND, verbose=False, diffuse=True)  # Generate the mu grid for the slab, this will be used to compute J_scat
            for i in range(ND):
                for m in range(0, self.slab_in.NM):
                    mu = self.slab_in.mu_values[m]
                    w_mu = self.slab_in.mu_weights[m]
                    for n in range(0, self.slab_in.NL):
                        x = self.x_values[n]
                        w_x = self.x_weights[n]
                        # Fetch the appropriate incident intensity
                        I_inc = self.slab_in.get_boundary_radiation(mu)  # Simple Gaussian profile
                        # Attenuation by optical depth
                        tau_eff = (self.slab_in.tau_max - self.slab_in.tau[i]) / mu * self.phi[n]
                        I_attenuated = I_inc * np.exp(-tau_eff)
                        J_inc[i] += I_attenuated * self.phi[n] * w_mu * w_x / 2.0  # Divide by 2 for J    

            self.J_scatter = J_inc
            del(J_inc)

        def compute_S_line(self, max_iter = 1000, tol = 1e-6, global_x_grid = None):
            # Use slab global grid if not provided
            if global_x_grid is None:
                if getattr(self.slab_in, "x_values", None) is None:
                    raise RuntimeError("compute_S_line: global_x_grid not provided and slab.x_values not set. Call slab.global_x_grid(lines) first.")
                global_x_grid = self.slab_in.x_values

            ND = self.slab_in.ND
            self.S_line = np.copy(self.B)

            # ensure mu grid exists (use existing slab.NM if set, otherwise default 8)
            N_mu = self.slab_in.NM if (self.slab_in.NM is not None) else 8
            self.slab_in.mu_grid(N_mu, verbose=False, diffuse=True)

            # Evaluate this line profile on the global x-grid and normalize using slab weights
            self.compute_phi_x(global_x_grid)   # fills self.phi_x at global_x_grid points
            if getattr(self.slab_in, "x_weights", None) is None:
                dx = global_x_grid[1] - global_x_grid[0]
                self.slab_in.x_weights = np.ones_like(global_x_grid) * dx
            norm = np.sum(self.phi_x * self.slab_in.x_weights)
            if norm != 0.0:
                phi_global = self.phi_x / norm
            else:
                phi_global = self.phi_x.copy()

            # Main Lambda Iteration loop (keeps structure but uses global grid & weights)
            for iteration in range(max_iter):
                J = np.zeros(ND)
                for m in range(0, self.slab_in.NM):
                    mu = self.slab_in.mu_values[m]
                    w_mu = self.slab_in.mu_weights[m]
                    for l in range(0, len(global_x_grid)):
                        w_x = self.slab_in.x_weights[l]
                        # use normalized phi on global grid and include line opacity factor self.k
                        tau_lambda = self.slab_in.tau * (self.k * phi_global[l] + self.r)

                        # Outwards
                        I_lambda = sc_2nd_order(tau_lambda, self.S_line, mu, 0.0)
                        J += I_lambda[0] * (0.5 * w_mu * phi_global[l] * w_x)

                        # Inwards
                        I_lambda = sc_2nd_order(tau_lambda, self.S_line, -mu, 0.0)
                        J += I_lambda[0] * (0.5 * w_mu * phi_global[l] * w_x)

                # update (basic Lambda iteration)
                dS = self.epsilon * self.B + (1. - self.epsilon) * J - self.S_line
                max_dS = np.max(np.abs(dS))
                self.S_line += dS
                if max_dS < tol:
                    print(f"Convergence achieved after {iteration} iterations with max dS = {max_dS:.2e}")
                    return
            # if not converged, leave S_line as last estimate
            print("compute_S_line: did not converge within max_iter")


    def compute_tau(self):
    # As the most robust method we will use log-spaced grid on both sides.
    # First we check the total ND, and make sure it's odd
        if self.ND % 2 == 0:
            self.ND += 1  # Make it odd
            
        # Create log-spaced grid for first half
        half_ND = self.ND // 2 + 1
        tau_first_half = np.logspace(-3, np.log10(self.tau_max / 2.0), half_ND)
        # Create log-spaced grid for second half
        # A bit of cheating because we want to get exact total tau_max
        tau_second_half = self.tau_max + tau_first_half[0] - tau_first_half[::-1]
        # Put these two together
        self.tau = np.concatenate((tau_first_half, tau_second_half[1:]))

    def global_x_grid(self, lines):
        # Build a global x-grid by concatenating all line local x-grids,
        # storing each line's local x values, and computing correct
        # non-uniform integration weights for the global grid.
        all_x = []
        for line in lines:
            # Ensure local x-grid exists on the line
            if line.x_grid is None:
                line.local_x_grid()
            # Keep an explicit copy of the local x-grid on the line
            line.local_x_values = np.array(line.x_grid, copy=True)
            all_x.append(line.local_x_values)

        if len(all_x) == 0:
            # No lines -> empty global grid
            self.x_values = np.array([])
            self.x_weights = np.array([])
            return

        # Concatenate, sort and take unique values (avoid duplicate x points)
        concatenated = np.concatenate(all_x)
        self.x_values = np.unique(np.sort(concatenated))

        # Compute non-uniform integration weights for the global x-grid.
        # Use the midpoint/trapezoidal style: w_i = (x_{i+1} - x_{i-1})/2
        x = self.x_values
        nx = x.size
        if nx == 0:
            self.x_weights = np.array([])
        elif nx == 1:
            self.x_weights = np.array([1.0])
        else:
            dx = np.empty_like(x)
            dx[0] = 0.5 * (x[1] - x[0])
            dx[-1] = 0.5 * (x[-1] - x[-2])
            if nx > 2:
                dx[1:-1] = 0.5 * (x[2:] - x[:-2])
            self.x_weights = dx

        # For each line save mapping from its local x-grid to the global grid
        for line in lines:
            # Map local x points to indices in the global grid. Use searchsorted
            # and clamp to valid indices to be robust to floating rounding.
            idx = np.searchsorted(self.x_values, line.local_x_values, side='left')
            idx[idx >= self.x_values.size] = self.x_values.size - 1
            # In rare cases searchsorted may point to a neighbour if values
            # differ by tiny eps; ensure nearest match (helps robustness).
            # Replace indices where the matched global value is farther than
            # the previous neighbor by checking distances.
            # (This is a lightweight correction; it won't change exact matches.)
            for j, ii in enumerate(idx):
                gval = self.x_values[ii]
                # check previous neighbor
                if ii > 0:
                    prev = self.x_values[ii - 1]
                    if abs(prev - line.local_x_values[j]) < abs(gval - line.local_x_values[j]):
                        idx[j] = ii - 1

            line.global_x_idx = idx
            line.global_x_values = self.x_values[idx]
            line.global_x_weights = self.x_weights[idx]

    def compute_phi(self, lines, correct_normalization=True, correction_extent=30.0):
        """Compute total line profile on global grid and normalize per-line profiles.
        
        Parameters:
        -----------
        lines : list
            List of Line objects.
        correct_normalization : bool
            If True, compute normalization correction using wider x-range to
            account for profile truncation. Eliminates sub-percent errors.
        correction_extent : float
            Extent for normalization correction evaluation (±extent from line center).
            Default 30.0 ensures normalization error <0.0001 for most profiles.
        """
        # We will compute the total line profile phi at each point in the global x grid by summing the contributions from all lines
        self.phi = np.zeros_like(self.x_values)
        for line in lines:
            # Evaluate the line profile at the global x-grid
            line.compute_phi_x(self.x_values)

            # Compute normalization on the global grid using the global x weights
            if getattr(self, 'x_weights', None) is None or self.x_weights.size != self.x_values.size:
                raise RuntimeError("Global x_weights not set or size mismatch. Call global_x_grid() first.")
            global_norm = np.sum(line.phi_x * self.x_weights)
            
            # Apply normalization correction if requested
            if correct_normalization:
                # Compute integral over wider range to get true normalized integral
                # This corrects for truncation of profile wings
                x_wide = np.linspace(line.line_center - correction_extent,
                                     line.line_center + correction_extent, 501)
                # Use simple trapezoidal rule with many points for accurate correction
                line.compute_phi_x(x_wide, type="voigt" if hasattr(line, 'a') else "gaussian")
                dx_wide = x_wide[1] - x_wide[0]  # uniform spacing for trapezoidal
                true_norm = np.trapz(line.phi_x, dx=dx_wide)
                
                # Correction factor: ratio of true integral to truncated integral
                correction_factor = true_norm / global_norm if global_norm != 0.0 else 1.0
                global_norm_corrected = global_norm * correction_factor
            else:
                global_norm_corrected = global_norm
            
            line.global_norm = global_norm_corrected
            
            # Re-evaluate phi_x on global grid after correction
            line.compute_phi_x(self.x_values)

            # Store a normalized version of phi on the global grid to avoid confusion
            if global_norm_corrected != 0.0:
                line.phi_x_global = line.phi_x / global_norm_corrected
            else:
                line.phi_x_global = line.phi_x.copy()

            # Add to total profile using the normalized per-line profile
            self.phi += line.k * line.phi_x_global  # contribution weighted by opacity ratio k
        self.phi = self.phi / np.sum(self.phi * self.x_weights)
    
    def mu_grid(self, N_mu, verbose=False, diffuse=False):
        self.NM = N_mu
        self.mu_values, self.mu_weights = np.polynomial.legendre.leggauss(N_mu)  # Gauss-Legendre quadrature for mu grid
        if diffuse:
            mu_crit = 0.0
        else:
            mu_crit = (1.0 - (const.R_sun.value**2.0 / (const.R_sun.value + self.H)**2.0))**0.5
        if verbose:
            print(f"Critical mu for illumination: {mu_crit}")
    

        # Now shift mu values to account for the critical mu, i.e. to pertain to the range [mu_crit, 1.0]
        self.mu_values = 0.5 * (self.mu_values + 1.0) * (1.0 - mu_crit) + mu_crit
        self.mu_weights = 0.5 * self.mu_weights * (1.0 - mu_crit)
        self.mu_weights /= np.sum(self.mu_weights)/(1.0-mu_crit)  # Normalize weights
        if verbose:
            print(f"Mu values: {self.mu_values}")
            print(f"Mu weights: {self.mu_weights}")

    def get_boundary_radiation(self, mu):
        # Here we are going to write a function that relates the mu in the slab referent frame to the outgoing 
        # limb darkening emerging from the solar surface. For the moment we will assume it is wavelength independent.     
        factor = 1.0 - ((1.0 - mu**2.0) * const.R_sun.value**2.0 / (const.R_sun.value + self.H)**2.0)
        if factor < 0.0:
            return 0.0
        else:
            mu_0 = factor**0.5
            # Simple linear limb darkening law
            # For more accuarate modeling we can use Claret coefficients for V filter (https://ui.adsabs.harvard.edu/abs/2000A&A...363.1081C/abstract)
            # Plot how this below looks like to make sure it goes from 1.0 at mu=1.0 to ~0.4 at mu=0.0 (or so)
            I_0 = 1 - 0.5311 * (1 - mu_0**0.5) + 0.0545 * (1 - mu_0) - 0.7301 * (1 - mu_0**1.5) + 0.4053 * (1 - mu_0**2)
            #I_0 = 0.4 + 0.6 * mu_0
            return I_0

    # Define formal solution for the emergent spectrum that lines can call for their source function
    def formal_solution(self, lines, mu, boundary_condition):
        """Compute emergent intensity spectrum at given observation angle.
        
        Parameters:
        -----------
        lines : list
            List of Line objects.
        mu : float
            Cosine of observation angle (mu=1.0 is vertical).
        boundary_condition : float
            Boundary condition intensity at top of slab.
        """
        # Setup: build global x-grid and compute per-line profiles
        self.S = np.copy(self.B)
        self.global_x_grid(lines)  # Build global x grid and compute weights
        self.compute_phi(lines, correct_normalization=True)
        
        I_emergent = np.zeros(len(self.x_values))
        
        # For each frequency point on the global x-grid:
        for i in range(len(self.x_values)):
            # Accumulate total optical depth from all lines at this frequency
            tau_lambda = np.zeros_like(self.tau)
            # Build numerator/denominator for composite source S_nu(depth)
            numer = np.zeros_like(self.tau)   # depth array
            denom = 0.0                       # scalar (sum of k*phi at this nu)

            for line in lines:
                # pick normalized per-line profile on global grid
                if hasattr(line, "phi_x_global"):
                    phi_use = line.phi_x_global
                else:
                    phi_use = line.phi_x.copy()
                    if getattr(self, "x_weights", None) is None:
                        dx = self.x_values[1] - self.x_values[0]
                        self.x_weights = np.ones_like(self.x_values) * dx
                    norm = np.sum(phi_use * self.x_weights)
                    if norm != 0.0:
                        phi_use = phi_use / norm

                # tau contribution (depth array)
                tau_contrib = self.tau * (line.k * phi_use[i] + line.r)
                tau_lambda += tau_contrib

                # component contribution to composite source: k * phi(nu) * S_line(depth)
                # require line.S_line to exist (line source per depth). fallback to B if missing.
                S_comp = getattr(line, "S_line", None)
                if S_comp is None:
                    S_comp = self.B
                numer += (line.k * phi_use[i]) * S_comp
                denom += (line.k * phi_use[i])

            # form composite S_nu(depth). if denom==0 fallback to slab B
            if denom != 0.0:
                S_nu = numer / denom
            else:
                S_nu = self.B

            # Solve radiative transfer for combined optical depth using frequency-dependent S_nu
            I = sc_2nd_order(tau_lambda, S_nu, mu, boundary_condition)
            I_emergent[i] = I[0, 0]  # Extract emergent intensity at top of slab (tau=0)
        
        self.I = I_emergent
        return I_emergent


if __name__ == "__main__":
   
    # Define the input parameters
    ND = 81
    tau_max = 1e3
    epsilon = np.ones(ND) * 5e-3
    B = np.ones(ND) * 5.0
    H = 80000.0 # Height of the slab above the solar surface in kilometers

    # Create the slab instance
    slab = Slab(ND, tau_max, epsilon, B, H)
    # Create the line instances
    line1 = slab.Line(81, line_center=0.0, a=0.1, k=1.0, r=0.0, slab_in=slab)  # Example line, we will need to pass the slab instance to the line   
    line2 = slab.Line(81, line_center=3.2, a=0.25, k=8.0, r=0.0, slab_in=slab)  # Another example line
    lines = [line1, line2]  
    slab.global_x_grid(lines)
    line1.compute_S_line(max_iter=1000, tol=1e-6, global_x_grid=slab.x_values)
    line2.compute_S_line(max_iter=1000, tol=1e-6, global_x_grid=slab.x_values)
    # Compute the source function for each line
    S = slab.formal_solution(lines, mu = 1.0, boundary_condition = 1.0)
    # Plot the intensity
    plt.figure(figsize=(10, 6))
    plt.plot(slab.x_values, S, label='Intensity')
    plt.xlabel('x (Doppler widths)')
    plt.ylabel('Intensity')
    plt.title('Emergent Intensity vs xe')
    plt.grid()
    plt.savefig('intensity_vs_x'+str(time.time())+'.png')