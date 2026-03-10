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
        self.r = 0.0
        # More parameters:
        self.mu_values = None
        self.mu_weights = None
        self.x_values = None
        self.x_weights = None

        self.compute_tau()  # Compute the tau grid, this function can also be used externally

    class Line:
        def __init__(self, NL, line_center, a, k, slab_in):
            self.line_center = line_center
            self.a = a # Damping parameter
            self.k = k # Ratio of the second line to the first line
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
        
        def local_x_grid(self, extent=6.0):
            """Generate local x-grid for this line.
            
            Parameters:
            -----------
            extent : float
                Distance from line center to grid boundaries (±extent).
                Default 15.0 to capture full profile without truncation errors.
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

        def compute_S_line(self, max_iter = 1000, tol = 1e-6, global_x_grid = None, type = "voigt"):
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
            self.compute_phi_x(global_x_grid, type = type)  
            print(f"Computed line profile for global x-grid with type: {type}") # fills self.phi_x at global_x_grid points
            if getattr(self.slab_in, "x_weights", None) is None:
                dx = global_x_grid[1] - global_x_grid[0]
                self.slab_in.x_weights = np.ones_like(global_x_grid) * dx
            norm = np.sum(self.phi_x * self.slab_in.x_weights)
            if norm != 0.0:
                phi_global = self.phi_x / norm
            else:
                phi_global = self.phi_x.copy()

            # Main Lambda Iteration loop (keeps structure but uses global grid & weights)
            # Let's try ALO
            for iteration in tqdm(range(max_iter)):
                J = np.zeros(ND)
                L = np.zeros(ND) # Lambda operator
                for m in range(0, self.slab_in.NM):
                    mu = self.slab_in.mu_values[m]
                    w_mu = self.slab_in.mu_weights[m]
                    for l in range(0, len(global_x_grid)):
                        w_x = self.slab_in.x_weights[l]
                        # use normalized phi on global grid and include line opacity factor self.k
                        tau_lambda = self.slab_in.tau * (self.k * phi_global[l] + self.slab_in.r)

                        # Outwards
                        I_lambda = sc_2nd_order(tau_lambda, self.S_line, mu, 0.0)
                        J += I_lambda[0] * (0.5 * w_mu * phi_global[l] * w_x)
                        L += I_lambda[1] * (0.5 * w_mu * phi_global[l] * w_x)

                        # Inwards
                        # use top-side illumination for inward ray
                        top_bc = self.slab_in.get_boundary_radiation(mu)
                        I_lambda = sc_2nd_order(tau_lambda, self.S_line, -mu, 0.0)
                        J += I_lambda[0] * (0.5 * w_mu * phi_global[l] * w_x)
                        L += I_lambda[1] * (0.5 * w_mu * phi_global[l] * w_x)

                # update (basic Lambda iteration)
                dS = (self.epsilon * self.B + (1. - self.epsilon) * J - self.S_line)/ (1.0 - (1.0 - self.epsilon) * L)
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
        boundary_condition : float or array
            Boundary condition intensity at top of slab. If float, same for all frequencies; if array, per frequency.
        """
        # Setup: build global x-grid and compute per-line profiles
        self.S = np.copy(self.B)
        self.global_x_grid(lines)  # Build global x grid and compute weights
        self.compute_phi(lines, correct_normalization=False)
        
        I_emergent = np.zeros(len(self.x_values))
        
        # Handle boundary_condition
        if np.isscalar(boundary_condition):
            bc = np.full(len(self.x_values), boundary_condition)
        else:
            bc = np.asarray(boundary_condition)
            if bc.shape != (len(self.x_values),):
                raise ValueError("boundary_condition array must match x_values length")
        
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
                tau_contrib = self.tau * (line.k * phi_use[i] + self.r)
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
            I = sc_2nd_order(tau_lambda, S_nu, mu, bc[i])
            I_emergent[i] = I[0, 0]  # Extract emergent intensity at top of slab (tau=0)
        
        self.I = I_emergent
        return I_emergent

    def composite_S(self, lines):
        """Compute composite source function S_nu(depth) on global x-grid.
        
        Parameters:
        -----------
        lines : list
            List of Line objects.
        """
        self.global_x_grid(lines)  # Build global x grid and compute weights
        self.compute_phi(lines, correct_normalization=False)

        S_nu_grid = np.zeros((len(self.tau), len(self.x_values)))  # depth x frequency array

        for i in range(len(self.x_values)):
            numer = np.zeros_like(self.tau)   # depth array
            denom = 0.0                       # scalar (sum of k*phi at this nu)

            for line in lines:
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

                S_comp = getattr(line, "S_line", None)
                if S_comp is None:
                    S_comp = self.B
                numer += (line.k * phi_use[i]) * S_comp
                denom += (line.k * phi_use[i])

            if denom != 0.0:
                S_nu_grid[:, i] = numer / denom
            else:
                S_nu_grid[:, i] = self.B

        return S_nu_grid


    def see_source_function(self, lines, slab_intensity):
        """Update source functions for all lines using current slab intensity, then update slab intensity.
        
        Parameters:
        -----------
        lines : list
            List of Line objects.
        slab_intensity : array
            Current emergent intensity spectrum of the slab on global x-grid.
        """
        for line in lines:
            line.compute_S_line(max_iter=1000, tol=1e-6, global_x_grid=self.x_values, boundary_condition=1.0)
        # Update slab intensity using the new S_line of each line
        self.I = self.formal_solution(lines, mu=1.0, boundary_condition=1.0)

    def iterate_coupled_lines(self, lines, max_iter=40, tol=1e-4, verbose=False, omega=1.0):
        """
        Iterate lines and slab consistently so lines "see" each other through the slab intensity.

        Algorithm (per outer iteration):
        - Ensure a common global x-grid and per-line normalized profiles exist.
        - Build current composite S_nu(depth) from line.S_line (fallback to B).
        - For all mu and frequency points solve formal RT once per (mu,nu) using composite S_nu to get I(depth,mu,nu).
        - For each line j accumulate J_j(depth) = 0.5 * sum_mu sum_n w_mu w_x phi_j(nu) * I(depth,mu,nu).
        - Update each line.S_line using ALI (diagonal lambda operator) if available, otherwise simple LI:
            S_new = [epsilon B + (1-epsilon) * J_nonlocal] / [1 - (1-epsilon) * Lambda_star]
          where J_nonlocal = J - Lambda_star * S_old.
        - Recompute composite S_nu and repeat until convergence.

        Returns dict with final composite S_nu grid, emergent intensity, per-line S_line, and rel history.
        """
        # prepare global grid & per-line normalized profiles
        self.global_x_grid(lines)
        self.compute_phi(lines, correct_normalization=False)
        # ensure x_weights present
        if getattr(self, "x_weights", None) is None or self.x_weights.size != self.x_values.size:
            dx = self.x_values[1] - self.x_values[0]
            self.x_weights = np.ones_like(self.x_values) * dx

        # ensure mu grid exists
        if getattr(self, "mu_values", None) is None or getattr(self, "mu_weights", None) is None:
            self.mu_grid(self.NM if self.NM is not None else 8, verbose=False, diffuse=True)

        Nfreq = len(self.x_values)
        ND = len(self.tau)
        Nmu = len(self.mu_values)

        # Ensure each line has phi_x_global and initialize S_line if missing
        for line in lines:
            if not hasattr(line, "phi_x_global"):
                line.compute_phi_x(self.x_values)
                norm = np.sum(line.phi_x * self.x_weights)
                line.phi_x_global = line.phi_x / norm if norm != 0.0 else line.phi_x.copy()
            if getattr(line, "S_line", None) is None:
                line.S_line = np.copy(self.B)

        # Using simple Lambda iteration only: no precomputed Lambda_star or ALI acceleration.

        rel_history = []
        for outer in tqdm(range(max_iter)):
            # build current composite S_nu on global grid (depth x freq)
            S_nu = self.composite_S(lines)  # uses current line.S_line internally

            # allocate J arrays per line (depth)
            J_lines = [np.zeros(ND) for _ in lines]

            # Loop over angles and frequencies once, using composite S_nu for formal solution
            for m in range(Nmu):
                mu = self.mu_values[m]
                w_mu = self.mu_weights[m]
                for l in range(Nfreq):
                    w_x = self.x_weights[l]
                    # total opacity at this freq: sum over lines (k*phi + r)
                    tau_lambda = np.zeros_like(self.tau)
                    for line in lines:
                        phi_l = line.phi_x_global[l]
                        tau_lambda += self.tau * (line.k * phi_l + self.r)
                    # formal solution for this (mu,nu) with frequency-dependent S_nu[:,l]
                    # include both outward and inward rays so lines see the correct illumination
                    # outward ray (from bottom, bottom boundary assumed zero)
                    I_out = sc_2nd_order(tau_lambda, S_nu[:, l], mu, 0.0)
                    I_out_depth = I_out[0]
                    # inward ray (from top) uses actual top illumination
                    top_bc = self.get_boundary_radiation(mu)
                    I_in = sc_2nd_order(tau_lambda, S_nu[:, l], -mu, 0.0)
                    I_in_depth = I_in[0]

                    # sum contributions from both directions
                    I_depth_sum = I_in_depth + I_in_depth

                    # accumulate J for each line using its phi at this frequency (use 1/2 when summing ±μ)
                    for j, line in enumerate(lines):
                        phi_j = line.phi_x_global[l]
                        J_lines[j] += 0.5 * w_mu * w_x * phi_j * I_in_depth
                        J_lines[j] += 0.5 * w_mu * w_x * phi_j * I_out_depth  
            # Now update each line's S_line using ALI / diagonal ALO if possible
            max_rel = 0.0
            tiny = 1e-20  # small threshold to avoid division by zero
            for j, line in enumerate(lines):
                J = J_lines[j]
                S_old = line.S_line

                # Simple LI update instead of ALI
                S_new = line.epsilon * line.B + (1.0 - line.epsilon) * J

                # under-relaxation
                S_updated = S_old + omega * (S_new - S_old)
                # measure relative change
                rel = np.max(np.abs((S_updated - S_old) / np.where(np.abs(S_updated) > tiny, S_updated, tiny)))
                max_rel = max(max_rel, rel)

                # store update
                line.S_line = S_updated

            rel_history.append(max_rel)
            if verbose:
                print(f"iterate_coupled_lines iter {outer:3d} max_rel={max_rel:.3e}")

            # update composite S and emergent intensity for diagnostics / next iteration
            S_nu = self.composite_S(lines)
            I_emergent = self.formal_solution(lines, mu=1.0, boundary_condition=0.0)

            # check convergence
            if max_rel < tol:
                if verbose:
                    print(f"iterate_coupled_lines converged after {outer} iterations (rel={max_rel:.3e})")
                break
        else:
            if verbose:
                print("iterate_coupled_lines: did not converge within max_iter")

        # final outputs
        final_S_nu = self.composite_S(lines)
        final_I = self.formal_solution(lines, mu=1.0, boundary_condition=0.0)
        print("Final emergent intensity:", final_I)
        return {
            "S_nu": final_S_nu,
            "I_emergent": final_I,
            "S_lines": [line.S_line.copy() for line in lines],
            "rel_history": np.array(rel_history)
        }

    # We should create a function that:
        # computes the source function for each line 
        # to do that, it needs to compute the J
        # to compute J, it needs to compute the emergent intensity from the slab (sc_2nd_order)
        # for emergent intensity, it shall use tau_lambda =+ tau * (k*phi) for each line + slab's r
        # J is then using line's phi normalized on the global grid
        # updates line source function using LI for each line
        # using updated line source functions to compute slab source function
        # slab source function is then used to compute emergent intensity again, and the process is repeated until convergence
    
    def lambda_iter_S(self, lines, max_iter=1000, tol=1e-6):
        self.global_x_grid(lines)
        self.compute_phi(lines, correct_normalization=False)
        # ensure x_weights present
        if getattr(self, "x_weights", None) is None or self.x_weights.size != self.x_values.size:
            dx = self.x_values[1] - self.x_values[0]
            self.x_weights = np.ones_like(self.x_values) * dx

        # ensure mu grid exists
        if getattr(self, "mu_values", None) is None or getattr(self, "mu_weights", None) is None:
            self.mu_grid(self.NM if self.NM is not None else 8, verbose=False, diffuse=True)

        Nfreq = len(self.x_values)
        ND = len(self.tau)
        Nmu = len(self.mu_values)
        # Ensure each line has phi_x_global and initialize S_line if missing
        for line in lines:
            if not hasattr(line, "phi_x_global"):
                line.compute_phi_x(self.x_values)
                norm = np.sum(line.phi_x * self.x_weights)
                line.phi_x_global = line.phi_x / norm if norm != 0.0 else line.phi_x.copy()
            if getattr(line, "S_line", None) is None:
                line.S_line = np.copy(self.B)
        for iteration in tqdm(range(max_iter)):
            S_nu = self.composite_S(lines)
            J_lines = [self.compute_J_line(line, S_nu) for line in lines]
            for m in range(Nmu):
                mu = self.mu_values[m]

            


def plot_help(slab, lines, result=None, max_iter=2000, tol=1e-6, save_prefix='', show=True):
    """Produce the standard set of diagnostic plots (same as the notebook cell):
    - emergent spectrum
    - composite S_nu heatmap
    - per-line S_line vs tau
    - convergence history
    - per-line contribution stackplot at mid depth

    Parameters
    ----------
    slab : Slab
        Slab instance
    lines : list
        List of Line instances
    result : dict, optional
        Result returned by slab.iterate_coupled_lines(). If None, the function
        will run iterate_coupled_lines() with provided max_iter/tol.
    max_iter, tol : passed to iterate_coupled_lines if result is None
    save_prefix : str
        Prefix for saved PNG filenames (default: '')
    show : bool
        If True, call plt.show() after plotting (default True)
    """
    import matplotlib.pyplot as _plt
    import numpy as _np

    if result is None:
        result = slab.iterate_coupled_lines(lines, max_iter=max_iter, tol=tol, verbose=False)

    S_nu = result["S_nu"]            # shape (ND, Nfreq)
    # fetch I_emergent without using boolean truth-testing on numpy arrays
    I_emergent = result.get("I_emergent", None)
    S_lines = result["S_lines"]      # list of per-line S_line (each length ND)
    rel_history = result.get("rel_history", None)

    x = slab.x_values
    tau = slab.tau

    # 1) Emergent spectrum
    try:
        _plt.figure(figsize=(8,4))
        _plt.plot(x, I_emergent, lw=2)
        _plt.xlabel("x (Doppler units)")
        _plt.ylabel("Emergent Intensity")
        _plt.title("Emergent Spectrum (iterate_coupled_lines)")
        _plt.grid(True)
        _plt.tight_layout()
        if save_prefix is not None:
            _plt.savefig(f'{save_prefix}emergent_spectrum.png', dpi=150)
        if show:
            _plt.show()
        else:
            _plt.close()
    except Exception:
        pass

    # 2) Composite source S_nu heatmap (depth x frequency)
    try:
        _plt.figure(figsize=(8,5))
        extent = [x[0], x[-1], tau[0], tau[-1]]
        _plt.imshow(S_nu, origin='lower', aspect='auto', extent=extent, cmap='viridis')
        _plt.colorbar(label='S_nu')
        _plt.xlabel("x (Doppler units)")
        _plt.ylabel("Optical depth (tau)")
        _plt.title("Composite S_nu (depth x frequency)")
        _plt.tight_layout()
        if save_prefix is not None:
            _plt.savefig(f'{save_prefix}composite_S_nu_heatmap.png', dpi=150)
        if show:
            _plt.show()
        else:
            _plt.close()
    except Exception:
        pass

    # 3) Per-line source functions vs tau (use S_lines from result to be consistent)
    try:
        _plt.figure(figsize=(6,4))
        for i, S_line in enumerate(S_lines):
            _plt.semilogy(_np.log10(tau), S_line, label=f'Line {i+1}')
        # overlay composite S at each line center on the same axes
        for i, line in enumerate(lines):
            idx = int(_np.argmin(_np.abs(x - line.line_center)))
            try:
                S_comp = S_nu[:, idx]
                _plt.semilogy(_np.log10(tau), S_comp, linestyle='--', lw=1.6, label=f'Composite @ x~{line.line_center:.2g}')
            except Exception:
                # if S_nu not available for some reason, skip
                pass

        # Planck function reference
        if _np.size(slab.B) == slab.ND:
            _plt.semilogy(_np.log10(tau), slab.B, ':k', label='Planck B (depth)')
        else:
            _plt.semilogy(_np.log10(tau), _np.mean(slab.B) * _np.ones_like(tau), ':k', label='Planck B (mean)')

        _plt.xlabel("Optical depth (tau)")
        _plt.ylabel("S_line / Composite S")
        _plt.title("Per-line Source Functions and Composite S at line centers")
        _plt.legend(loc='best')
        _plt.grid(True)
        _plt.tight_layout()
        if save_prefix is not None:
            _plt.savefig(f'{save_prefix}per_line_S_vs_tau.png', dpi=150)
        if show:
            _plt.show()
        else:
            _plt.close()
    except Exception:
        pass

    # 4) Relative convergence history
    try:
        if rel_history is not None and len(rel_history) > 0:
            _plt.figure(figsize=(5,3))
            _plt.semilogy(_np.arange(len(rel_history)), rel_history, marker='o')
            _plt.xlabel("Outer iteration")
            _plt.ylabel("max_rel")
            _plt.title("Convergence history")
            _plt.grid(True)
            _plt.tight_layout()
            if save_prefix is not None:
                _plt.savefig(f'{save_prefix}convergence_history.png', dpi=150)
            if show:
                _plt.show()
            else:
                _plt.close()
    except Exception:
        pass

    # 5) Per-line contribution at mid depth (stacked area) — using denom-based contributions
    try:
        # ensure phi_x_global exists for each line and is normalized
        for i, line in enumerate(lines):
            if not hasattr(line, 'phi_x_global') or line.phi_x_global.size != x.size:
                if hasattr(line, 'compute_phi_x'):
                    line.compute_phi_x(x)
                norm = _np.sum(line.phi_x * getattr(slab, 'x_weights', _np.ones_like(x) * (x[1] - x[0])))
                line.phi_x_global = (line.phi_x / norm) if norm != 0.0 else line.phi_x.copy()
            # print normalization diagnostic
            sphi = _np.sum(line.phi_x_global * getattr(slab, 'x_weights', _np.ones_like(x) * (x[1] - x[0])))
            print(f'Line {i+1} phi normalization (sum phi*xw): {sphi:.6g}')
        # build emissivities (eta = k * phi * S_line)
        eta_list = []
        for i, line in enumerate(lines):
            eta = (line.k * line.phi_x_global)[None, :] * _np.asarray(S_lines[i]).reshape((slab.ND, 1))
            eta_list.append(eta)
        # denom-based contributions (as in composite S construction)
        denom = _np.zeros_like(x)
        for line in lines:
            denom += line.k * line.phi_x_global
        denom_safe = _np.where(denom == 0.0, 1.0, denom)
        contribs = [eta / denom_safe[None, :] for eta in eta_list]
        mid = slab.ND // 2
        contrib_mid = _np.vstack([c[mid, :] for c in contribs])
        contrib_mid = _np.clip(contrib_mid, 0.0, None)
        # normalize across lines at each x to avoid tiny rounding errors
        sum_c = _np.sum(contrib_mid, axis=0)
        sum_c_safe = _np.where(sum_c == 0.0, 1.0, sum_c)
        contrib_mid_norm = contrib_mid / sum_c_safe
        labels = [f'Line {i+1} (x={lines[i].line_center})' for i in range(len(lines))]
        palette = ['cyan', 'lime', 'maroon', 'magenta', 'yellow', 'orange']
        _plt.figure(figsize=(10,4))
        _plt.stackplot(x, contrib_mid_norm, labels=labels, colors=[palette[i % len(palette)] for i in range(len(lines))], alpha=0.85)
        comp_mid = S_nu[mid, :]
        comp_scaled = (comp_mid - _np.min(comp_mid)) / (_np.max(comp_mid) - _np.min(comp_mid) + 1e-30)
        _plt.plot(x, comp_scaled, '--k', lw=2, label='Composite S (mid, scaled)')
        _plt.xlabel("x (Doppler units)")
        _plt.ylabel("Per-line contribution (stacked)")
        _plt.title("Per-line contributions (mid depth) with composite S overlay")
        _plt.legend(loc='upper right')
        _plt.grid(True)
        _plt.tight_layout()
        if save_prefix is not None:
            _plt.savefig(f'{save_prefix}per_line_contribs_stack_mid.png', dpi=150)
        if show:
            _plt.show()
        else:
            _plt.close()
    except Exception as _e:
        print('plot_help per-line contributions failed:', _e)
        pass

    # 6) Per-line contributions vs x (all depths overlay; avg & peak shown)
    try:
        # ensure contribs available (rebuild if necessary)
        try:
            contribs
        except NameError:
            eta_list = []
            for i, line in enumerate(lines):
                eta = (line.k * line.phi_x_global)[None, :] * _np.asarray(S_lines[i]).reshape((slab.ND, 1))
                eta_list.append(eta)
            denom = _np.zeros_like(x)
            for line in lines:
                denom += line.k * line.phi_x_global
            denom_safe = _np.where(denom == 0.0, 1.0, denom)
            contribs = [eta / denom_safe[None, :] for eta in eta_list]

        palette = ['cyan', 'lime', 'maroon', 'magenta', 'yellow', 'orange']
        _plt.figure(figsize=(11,6))
        decim = max(1, int(_np.ceil(slab.ND / 60)))
        depths_all = _np.arange(0, slab.ND, decim)
        for idx, contrib in enumerate(contribs):
            color = palette[idx % len(palette)]
            # many faint depth curves
            for d in depths_all:
                _plt.plot(x, contrib[d, :], color=color, alpha=0.08, linewidth=1)
            # average curve across sampled depths
            mean_curve = _np.mean(contrib[depths_all, :], axis=0)
            _plt.plot(x, mean_curve, color=color, lw=1.8, alpha=0.9, label=f'Line @{lines[idx].line_center} contribution (avg depths)')
            # peak depth curve (by integrated contribution)
            if getattr(slab, 'x_weights', None) is None:
                xw = _np.ones_like(x) * (x[1] - x[0])
            else:
                xw = slab.x_weights
            total_per_depth = _np.sum(contrib * xw[None, :], axis=1)
            depth_max = int(_np.argmax(total_per_depth))
            _plt.plot(x, contrib[depth_max, :], color=color, lw=2.6, alpha=1.0, linestyle='--', label=f'Line @{lines[idx].line_center} peak depth idx={depth_max}, tau={slab.tau[depth_max]:.2g}')

        # overlay composite S (mid depth) and Planck B reference
        mid = slab.ND // 2
        _plt.plot(x, S_nu[mid, :], '--k', lw=2.0, label='Composite S (mid depth)')
        _plt.plot(x, _np.mean(slab.B) * _np.ones_like(x), ':k', label='Planck B (mean)')
        _plt.xlabel('x (Doppler units)')
        _plt.ylabel('Per-line contribution to composite S')
        _plt.title('Per-line contributions vs x (all depths overlay; avg & peak shown)')
        _plt.legend(loc='upper right')
        _plt.grid(True)
        _plt.tight_layout()
        if save_prefix is not None:
            _plt.savefig(f'{save_prefix}per_line_contributions_all_depths_vs_x.png', dpi=150)
        if show:
            _plt.show()
        else:
            _plt.close()
    except Exception as _e:
        print('plot_help combined contributions failed:', _e)
        pass

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
    line1 = slab.Line(81, line_center=0.0, a=0.1, k=1.0, slab_in=slab)  # Example line, we will need to pass the slab instance to the line   
    line2 = slab.Line(81, line_center=3.2, a=0.2, k=8.0, slab_in=slab)  # Another example line
    #line3 = slab.Line(81, line_center=8.5, a= 0.5, k=3.0, slab_in=slab)
    lines = [line1, line2]
    slab.global_x_grid(lines)
    line1.compute_S_line(max_iter=2000, tol=1e-6, global_x_grid=slab.x_values)
    line2.compute_S_line(max_iter=2000, tol=1e-6, global_x_grid=slab.x_values)
    #line3.compute_S_line(max_iter=2000, tol=1e-6, global_x_grid=slab.x_values)
    # Compute the source function for each line
    S = slab.formal_solution(lines, mu = 1.0, boundary_condition = 1.0)
    # Plot the intensity
    plt.figure(figsize=(10, 6))
    plt.plot(slab.x_values, S, label='Intensity')
    # Plot Planck function B (use mean if B is depth-dependent) as reference
    plt.plot(slab.x_values, np.mean(slab.B) * np.ones_like(slab.x_values), ':k', label='Planck B')
    plt.xlabel('x (Doppler widths)')
    plt.ylabel('Intensity')
    plt.title('Emergent Intensity vs xe')
    plt.grid()
    plt.savefig('intensity_vs_x_k1_'+str(line1.k)+'_k2_'+str(line2.k)+'.png')


    # Plot the source function of each line and composite source function
    plt.figure(figsize=(10, 6))
    for line in lines:
        if hasattr(line, "S_line"):
            plt.semilogy(np.log10(slab.tau), line.S_line, label=f'Line at x={line.line_center}')
    composite_S = slab.composite_S(lines)
    plt.semilogy(np.log10(slab.tau), composite_S[:, slab.x_values.size//2], label='Composite S at line center', linestyle='--')
    plt.semilogy(np.log10(slab.tau), slab.B, ':k', label = 'Planck B')
    plt.xlabel('Optical Depth (tau)')
    plt.ylabel('Source Function')
    plt.title('Source Function vs Optical Depth')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.savefig('source_function_vs_tau_k1_'+str(line1.k)+'_k2_'+str(line2.k)+'.png')


    # Build composite source function grid (depth x frequency)
    S_nu_grid = slab.composite_S(lines)   # shape (ND, Nfreq)

    # Compute emergent intensity (formal solution) and store in slab.I
    I_emergent = slab.formal_solution(lines, mu = 1.0, boundary_condition = 1.0)

    def plot_line_and_composite(lines, slab, S_nu_grid, I_emergent):
        x = slab.x_values
        tau = slab.tau
        B = slab.B
        # ensure per-line normalized profile present
        for line in lines:
            if not hasattr(line, 'phi_x_global'):
                line.compute_phi_x(x)
                if getattr(slab, 'x_weights', None) is None:
                    dx = x[1] - x[0]
                    slab.x_weights = np.ones_like(x) * dx
                norm = np.sum(line.phi_x * slab.x_weights)
                line.phi_x_global = line.phi_x / norm if norm != 0.0 else line.phi_x.copy()

        # Precompute denom used in composite S
        denom = np.zeros_like(x)
        for line in lines:
            denom += line.k * line.phi_x_global
        denom_safe = np.where(denom == 0.0, 1.0, denom)

        # 1) For each line: emissivity vs x at selected depths, and S_line vs tau with Planck B
        depths_idx = [0, slab.ND // 2, slab.ND - 1]  # top, mid, bottom
        for i, line in enumerate(lines):
            # emissivity spectra at selected depths: eps(x) = k * phi(x) * S_line(depth)
            plt.figure(figsize=(8,5))
            for d in depths_idx:
                emiss = line.k * line.phi_x_global * line.S_line[d]
                plt.plot(x, emiss, label=f'depth idx {d}')
            plt.plot(x, np.mean(B) * np.ones_like(x), ':k', label='Planck B (mean)')
            plt.xlabel('x (Doppler units)')
            plt.ylabel('Emissivity ~ k phi(x) S_line(depth)')
            plt.title(f'Line {i+1} emissivity vs x (selected depths) — center={line.line_center}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'line_{i+1}_emissivity_vs_x.png', dpi=150)
            plt.close()

            # S_line vs tau with Planck B reference
            plt.figure(figsize=(6,5))
            plt.plot(tau, line.S_line, label=f'Line {i+1} S_line')
            # Plot Planck B; if B is depth-dependent plot full curve else mean
            if np.size(B) == slab.ND:
                plt.plot(tau, B, ':k', label='Planck B (depth)')
            else:
                plt.plot(tau, np.mean(B) * np.ones_like(tau), ':k', label='Planck B (mean)')
            plt.xscale('log')
            plt.xlabel('Optical depth (tau)')
            plt.ylabel('Source function')
            plt.title(f'Line {i+1} Source function vs tau')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'line_{i+1}_S_vs_tau.png', dpi=150)
            plt.close()

        # Emergent intensity spectrum
        plt.figure(figsize=(8,5))
        plt.plot(x, I_emergent, label='Emergent Intensity')
        plt.plot(x, np.mean(B) * np.ones_like(x), ':k', label='Planck B (mean)')
        plt.xlabel('x (Doppler units)')
        plt.ylabel('Intensity')
        plt.title('Emergent Intensity Spectrum')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('emergent_intensity_vs_x.png', dpi=150)
        plt.close()

        # 2) Composite source: S_nu vs x for selected depths, with Planck B
        plt.figure(figsize=(8,5))
        for d in depths_idx:
            plt.plot(x, S_nu_grid[d, :], label=f'Composite S at depth idx {d}')
        plt.plot(x, np.mean(B) * np.ones_like(x), ':k', label='Planck B (mean)')
        plt.xlabel('x (Doppler units)')
        plt.ylabel('Composite source S_nu (depth slice)')
        plt.title('Composite Source Function vs x (selected depths)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('composite_S_vs_x_selected_depths.png', dpi=150)
        plt.close()

        # 3) Composite source vs tau at line centers (for each line choose its center index)
        plt.figure(figsize=(6,5))
        for i, line in enumerate(lines):
            idx = np.argmin(np.abs(x - line.line_center))
            plt.plot(tau, S_nu_grid[:, idx], label=f'Composite S at x~{line.line_center:.2g}')
        if np.size(B) == slab.ND:
            plt.plot(tau, B, ':k', label='Planck B (depth)')
        else:
            plt.plot(tau, np.mean(B) * np.ones_like(tau), ':k', label='Planck B (mean)')
        plt.xscale('log')
        plt.xlabel('Optical depth (tau)')
        plt.ylabel('Composite source at selected x')
        plt.title('Composite S vs tau at line centers')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('composite_S_vs_tau_line_centers.png', dpi=150)
        plt.close()

        # 4) Per-line contribution to composite S for all depths (combined single figure)
        # contribution(depth,x) = (k * phi(x) * S_line(depth)) / denom(x)
        for i, line in enumerate(lines):
            contrib = (line.k * line.phi_x_global[None, :] ) * line.S_line[:, None]
            contrib /= denom_safe[None, :]  # safe division
        # Build per-line contribution arrays (depth x frequency) and plot them ON THE SAME figure
        contribs = []
        for line in lines:
            contrib = (line.k * line.phi_x_global[None, :]) * line.S_line[:, None]
            contrib /= denom_safe[None, :]  # safe division
            contribs.append(contrib)

        # Single combined figure: composite S heatmap + contour overlays of contributions
        plt.figure(figsize=(10, 6))
        im = plt.imshow(S_nu_grid, aspect='auto', origin='lower',
                        extent=[x[0], x[-1], tau[0], tau[-1]], cmap='plasma')
        cbar = plt.colorbar(im, label='Composite S_nu(depth, x)')
        # choose distinct contour colors for up to 6 lines
        palette = ['cyan', 'lime', 'maroon', 'magenta', 'yellow', 'orange']
        contour_levels = [0.05, 0.1, 0.25, 0.5]  # fractional contribution levels

        # Build explicit legend handles (one per line) to avoid relying on contour internals
        from matplotlib.lines import Line2D
        legend_handles = []
        for idx, contrib in enumerate(contribs):
            color = palette[idx % len(palette)]
            # draw contours; guard against any library differences
            try:
                CS = plt.contour(x, tau, contrib, levels=contour_levels, colors=[color],
                                 linewidths=1.0, alpha=0.9)
            except Exception:
                # fallback: draw a thin filled contourf for visibility
                plt.contourf(x, tau, contrib, levels=[0.0] + contour_levels, colors=[color], alpha=0.05)

            # create a proxy artist for legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2, label=f'Line {idx+1} contrib'))

        plt.xlabel('x (Doppler units)')
        plt.ylabel('Optical depth (tau)')
        plt.title('Composite S_nu (heatmap) with line contribution contours')
        plt.gca().set_yscale('linear')
        # add legend for contribution proxies + a proxy for heatmap
        heatmap_proxy = Line2D([0], [0], color='k', lw=2, linestyle='--', label='Composite S (example)')
        all_handles = [heatmap_proxy] + legend_handles
        plt.legend(handles=all_handles, loc='upper right')
        plt.tight_layout()
        plt.savefig('combined_line_contributions_depth_x.png', dpi=150)
        plt.close()

        # 5) Per-line contributions vs x for ALL depths (decimated curves to avoid overplotting)
        plt.figure(figsize=(11,6))
        decim = max(1, int(np.ceil(slab.ND / 60)))  # plot up to ~60 depth curves
        depths_all = np.arange(0, slab.ND, decim)
        for idx, contrib in enumerate(contribs):
            color = palette[idx % len(palette)]
            # plot many depth slices with low alpha
            for d in depths_all:
                plt.plot(x, contrib[d, :], color=color, alpha=0.08, linewidth=1)
            # overlay a thicker mean contribution curve for visibility
            mean_curve = np.mean(contrib[depths_all, :], axis=0)
            plt.plot(x, mean_curve, color=color, lw=1.5, alpha=0.9, label=f'Line @{lines[idx].line_center} contribution (avg depths)')

            # find depth where this line contributes the most (integrated over x with weights)
            if getattr(slab, "x_weights", None) is None:
                dx = x[1] - x[0]
                xw = np.ones_like(x) * dx
            else:
                xw = slab.x_weights
            # total fractional contribution per depth (weighted integral over x)
            total_per_depth = np.sum(contrib * xw[None, :], axis=1)
            depth_max = int(np.argmax(total_per_depth))
            # plot the contribution curve at that depth as a thick line
            plt.plot(x, contrib[depth_max, :], color=color, lw=2.5, alpha=1.0, linestyle = "--",
                     label=f'Line @{lines[idx].line_center} peak depth idx={depth_max}, tau={slab.tau[depth_max]:.2g}')

        # overlay composite S (choose mid depth) as dashed black curve for comparison
        mid = slab.ND // 2
        plt.plot(x, S_nu_grid[mid, :], '--k', lw=2.0, label='Composite S (mid depth)')
        # overlay Planck B reference (mean)
        plt.plot(x, np.mean(B) * np.ones_like(x), ':k', label='Planck B (mean)')
        plt.xlabel('x (Doppler units)')
        plt.ylabel('Source function')
        plt.title('Per-line contributions vs x (all depths overlay; avg & mid shown)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('per_line_contributions_all_depths_vs_x.png', dpi=150)
        plt.close()

        print("Saved plots: per-line emissivity, S_vs_tau, composite S vs x/tau, combined contribution heatmap, and per-line contributions (all depths).")
    plot_line_and_composite(lines, slab, S_nu_grid, I_emergent)

    # 20. 02. 2026.
    # Svaka linija treba da ima posebnu normu za phi koja ulazi u njen J (ne mora normirati na globalu)
    # Linije komuniciraju medjusobno kroz intenzitet slaba
    # tau grid slaba neka uzme centar prve linije
    # Lambda iteracija treba da koristi slab intenzitet, koji ce se menjati u zavisnosti od S_line svake linije, a ne samo B
    # tau_lambda je tau * (k*phi + r)
    # r je svojstvo slaba, ne linije 
    # Procedura treba da bude veca: 
    # 1) Linija vidi intenzitet slaba
    # 2) Linija racuna svoj J koristeci taj intenzitet i svoju phi (koja je normirana na lokalnom x-gridu)
    # 3) Linija update-uje svoj S_line koristeci J
    # 4) Slab update-uje svoj intenzitet koristeci S_line svih linija (ukupna funkcija izvora)
    # 5) Sanity check: posle prve iteracije funkcija izvora treba da bude <= B
    # 6) 