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

            # Inherited properties from the slab
            self.B = slab_in.B # Planck function
            self.epsilon = slab_in.epsilon # Thermalization parameter, same as the slab's epsilon
        
        def local_x_grid(self):
            start = self.line_center - 5.0  # Start of the local x grid
            end = self.line_center + 5.0    # End of the local x grid
            self.x_grid = np.linspace(start, end, self.NL)  # Local x grid for this line
            # Doesn't have to be equidistant

        def compute_phi_x(self, x, type="voigt"):
            if type == "voigt":
                z = x + 1j * self.a  # Complex argument for the Voigt profile
                self.phi_x = np.real(wofz(z)) / (np.sqrt(np.pi))
            elif type == "gaussian":
                self.phi_x = np.exp(-x**2) / np.sqrt(np.pi)  # Gaussian profile
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

        # Now we can compute the source function for the line on the global x grid, which will be used to compute the emergent spectrum.
        def compute_S_line(self, max_iter = 1000, tol = 1e-6):
            ND = self.slab_in.ND
            self.slab_in.mu_grid(self.slab_in.ND, verbose=False, diffuse=False)  # Generate the mu grid for the slab, this will be used to compute J_scat
            #self.J_diff = np.zeros((ND, self.NL))
            self.local_x_grid()
            self.compute_phi_x(self.x_grid)  # Compute the line profile at the local x grid, this will be used to compute the source function
            self.compute_weights(verbose=False)  # Compute the weights for the local x grid, this will be used to compute the source function
            self.S_line = np.copy(self.B)  # Initial guess for the source function, we can start with the Planck function
            # We will use LI method to iteratively solve for the source function.
            # We will use emergent spectrum to compute the mean intensity J, and then update the source function.
            for iteration in range(max_iter):
                self.J = np.zeros(ND)
                for m in range(0, self.slab_in.NM):
                    for l in range(len(self.x_grid)):
                        mu = self.slab_in.mu_values[m]
                        w_mu = self.slab_in.mu_weights[m]
                        x = self.x_grid[l]
                        w_x = self.x_weights[l]
                        tau_lambda = self.slab_in.tau * (self.phi_x[l]*self.k + self.r)  # Total optical depth at this frequency point, including contributions from the line and the continuum
                        # Outward intensity with positive mu
                        I_line = sc_2nd_order(tau_lambda, self.S_line, mu, 0.0)
                        #I_line = self.slab_in.formal_solution(self.S_line, mu, boundary_condition='outward')
                        self.J += I_line[0] * self.phi_x[l] * w_mu * w_x / 2.0

                        # Inward intensity with negative mu
                        I_line = sc_2nd_order(tau_lambda, self.S_line, -mu, 0.0)
                        #I_line = self.slab_in.formal_solution(self.S_line, -mu, boundary_condition='inward')
                        self.J += I_line[0] * self.phi_x[l] * w_mu * w_x / 2.0
                        
                # Sum J over frequency points
                # self.J = np.sum(self.J_diff*self.phi_x[None,:]*self.x_weights[None,:], axis=1)
                # Update the source function using the new J
                S_new = self.epsilon * self.B + (1 - self.epsilon) * self.J
                # Check for convergence
                if np.max(np.abs(S_new - self.S_line) / self.B) < tol:
                    print(f"Source function converged after {iteration} iterations.")
                    break
                self.S_line = S_new  # Update the source function for the next iteration
            else:
                print("Source function did not converge within the maximum number of iterations.")

            # New method: compute source function using emergent intensity on the slab global x-grid
        def compute_S_line_global(self, lines, max_iter=1000, tol=1e-6, verbose=False, N_mu=8):
            """
            Lambda-iteration over depth (tau), angle (mu) and frequency (x_local).
            Uses sc_2nd_order to obtain intensities at every depth for the total (global) tau_lambda
            computed from the full set of 'lines'. The local line source function is updated
            using the mean intensity constructed from those intensities (loop order: tau, mu, x).
            """
            slab = self.slab_in
            ND = slab.ND

            # Build global frequency grid and total profiles on it
            slab.global_x_grid(lines)
            slab.compute_phi(lines)  # sets each line.phi_x on slab.x_values
            global_x = slab.x_values

            # Ensure mu grid for radiative transfer
            slab.mu_grid(N_mu, verbose=False, diffuse=False)

            # Prepare this line's local grid, local phi and local weights (do not overwrite line.phi_x on global grid)
            self.local_x_grid()
            x_local = self.x_grid.copy()
            # local Voigt/Gauss profile (compute locally)
            phi_local = np.real(wofz(x_local + 1j * self.a)) / np.sqrt(np.pi)
            # simple uniform frequency weights on local grid normalized by profile
            xw = np.ones_like(x_local) * (x_local[-1] - x_local[0]) / len(x_local)
            xw = xw / np.sum(phi_local * xw)
            self.x_weights = xw

            # Map each local frequency to nearest index on global grid
            idx_map = np.array([np.argmin(np.abs(global_x - xv)) for xv in x_local], dtype=int)

            # Initial guess for the source function (per depth)
            S_curr = np.copy(self.B)

            # Pre-extract list reference for speed
            lines_list = lines

            for it in range(max_iter):
                J = np.zeros(ND)  # mean intensity per depth for this line
                # Loop order: mu -> x (local)
                for m in range(slab.NM):
                    mu = slab.mu_values[m]
                    w_mu = slab.mu_weights[m]
                    for lx in range(len(x_local)):
                        g = idx_map[lx]  # corresponding global frequency index
                        # total optical depth at this global frequency (sum over all lines + continuum r)
                        tau_lambda = slab.tau * np.sum([ (ln.phi_x[g] * ln.k + ln.r) for ln in lines_list ], axis=0)
                        # obtain intensities at all depths for this tau_lambda and current S_curr
                        I_res = sc_2nd_order(tau_lambda, S_curr, mu, 0.0)
                        # sc_2nd_order typically returns array-like; take first element if tuple/list
                        if isinstance(I_res, (tuple, list)):
                            I_all = np.asarray(I_res[0])
                        else:
                            I_all = np.asarray(I_res)
                        # accumulate J at all depths (angle and frequency integration), divide by 2 for angular averaging over both hemispheres
                        J += I_res[0] * phi_local[lx] * w_mu * xw[lx] / 2.0

                        I_res = sc_2nd_order(tau_lambda, S_curr, -mu, 0.0)
                        if isinstance(I_res, (tuple, list)):
                            I_all = np.asarray(I_res[0])
                        else:
                            I_all = np.asarray(I_res)       
                        J += I_res[0] * phi_local[lx] * w_mu * xw[lx] / 2.0 

                # Update source function (local line)
                S_new = self.epsilon * self.B + (1.0 - self.epsilon) * J

                # Convergence check (relative to Planck B)
                if np.max(np.abs(S_new - S_curr) / (np.abs(self.B) + 1e-30)) < tol:
                    if verbose:
                        print(f"compute_S_line_global: converged in {it} iterations")
                    S_curr = S_new
                    break

                S_curr = S_new

            else:
                if verbose:
                    print("compute_S_line_global: did not converge within max_iter")

            # store result
            self.S_line = S_curr
            return self.S_line
            

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

    def global_tau(self, lines):
        # We will compute the global tau grid by summing the contributions from all lines and the continuum
        self.tau_grid = np.zeros_like(self.tau)
        for line in lines:
            line.local_x_grid()  # Generate the local x grid for this line
            line.compute_phi_x(line.x_grid)  # Compute the line profile for this line at the local x grid
            self.tau_grid += self.tau * (line.phi_x * line.k + line.r)  # Add the contribution from this line, weighted by its opacity ratio r

    def global_x_grid(self, lines):
        # We will concatenate the local x grids from all lines and then sort them
        all_x = []
        for line in lines:
            line.local_x_grid()  # Generate the local x grid for this line
            all_x.append(line.x_grid)  # Append it to the list of all x grids
        
        # Concatenate and sort the global x grid
        self.x_values = np.sort(np.concatenate(all_x))
    
    def compute_phi(self, lines):
        # We will compute the total line profile phi at each point in the global x grid by summing the contributions from all lines
        self.phi = np.zeros_like(self.x_values)
        for line in lines:
            line.compute_phi_x(self.x_values)  # Compute the line profile for this line at the global x grid
            self.phi += line.r * line.phi_x  # Add the contribution from this line, weighted by its opacity ratio r
    
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
    def formal_solution(self, S, mu_ob, boundary_condition='outward'):
        line1 = self.Line(41, line_center=0.0, a=0.1, k=1.0, r=0.0, slab_in=self)  # Example line, we will need to pass the slab instance to the line
        line2 = self.Line(41, line_center=3.2, a=0.2, k=8.0, r=0.0, slab_in=self)  # Another example line
        self.global_x_grid([line1, line2])  # Make sure the global x grid is generated before we compute the emergent spectrum
        self.mu_grid(N_mu=8, verbose=False, diffuse=False)  # Generate the mu grid for the slab, this will be used to compute J_scat
        self.compute_phi([line1, line2])  # Compute the total line profile phi for all lines in the slab
        spectrum = np.zeros(len(self.x_values))
        self.S = np.copy(S)
        if boundary_condition == 'outward':
            for l in range(len(self.x_values)):
                # For each frequency point, compute the emergent intensity
                tau_lambda = self.tau * (line1.phi_x[l]*line1.k + line2.phi_x[l]*line2.k + line1.r + line2.r)  # Total optical depth at this frequency point, including contributions from both lines and the continuum
                I_emergent = sc_2nd_order(tau_lambda, self.S, mu_ob, boundary_condition)
                #print('!!!!!!!!', I_emergent[0].shape)
                spectrum[l] = I_emergent[0,0]  # Store the emergent intensity only, we don't need the lambda operator here
        if boundary_condition == 'inward':
            for l in range(len(self.x_values)):
                # For each frequency point, compute the inward intensity
                tau_lambda = self.tau * (line1.phi_x[l]*line1.k + line2.phi_x[l]*line2.k + line1.r + line2.r)
                I_inward = sc_2nd_order(tau_lambda, self.S, -mu_ob, boundary_condition)
                spectrum[l] = I_inward[0,0]  # Store the inward intensity only, we don't need the lambda operator here
        return spectrum  # Return the emergent spectrum
    

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
    line2 = slab.Line(81, line_center=3.2, a=0.2, k=8.0, r=0.0, slab_in=slab)  # Another example line
    lines = [line1, line2]  
    # Compute the source function for each line
    
    line1.compute_S_line()
    line2.compute_S_line()

    # Plot the source function for each line vs index
    plt.figure(figsize=(10, 6))
    plt.plot(slab.tau, line1.S_line, label='Line 1 Source Function')
    plt.plot(slab.tau, line2.S_line, label='Line 2 Source Function')
    plt.xscale('log')
    plt.xlabel('Optical Depth (tau)')
    plt.ylabel('Source Function (S)')
    plt.title('Source Function for Each Line')
    plt.legend()
    plt.grid()
    plt.show()


    slab.global_x_grid(lines)
    slab.compute_phi(lines)
    S_1 = line1.compute_S_line_global(lines, max_iter=1000, tol=1e-6, verbose=True, N_mu=8)
    S_2 = line2.compute_S_line_global(lines, max_iter=1000, tol=1e-6, verbose=True, N_mu=8)
    # Plot the source function for each line vs index
    plt.figure(figsize=(10, 6))
    plt.plot(slab.tau, S_1, label='Line 1 Source Function (Global)')
    plt.plot(slab.tau, S_2, label='Line 2 Source Function (Global)')
    plt.xscale('log')
    plt.xlabel('Optical Depth (tau)')
    plt.ylabel('Source Function (S)')
    plt.legend()
    plt.title('Source Function for Each Line (Global x-grid)')
    plt.grid()
    plt.show()