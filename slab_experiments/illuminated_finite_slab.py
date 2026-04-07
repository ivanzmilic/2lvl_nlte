import numpy as np 
import matplotlib.pyplot as plt
import astropy.units as units
import astropy.constants as const
from scipy.special import wofz
from tqdm import tqdm
from rtfunctions import one_full_fs, sc_2nd_order, calc_lambda_full, calc_lambda_monoc
import time

# Some basic description, so we know what we are doing: 
# This script solves the NLTE line formation for a 2-level atom in an illuminated finite slab.
# The slab is illuminated from one side by a radiation field with a given angular 
# and spectral distribution. The goal is to compute the emergent intensity and the source function 
# within the slab, taking into account both scattering and thermal emission.

# The flow is something like this / update 03/02/2026:
# In this version of the flow we want to have multiple spectral lines in a slab

# 1. Define epsilon, B, and other parameters that are common for all the lines 
# 
# 2  Here we also have to likely define the wavelength and angle grid. 
#    Keep in mind that normalization of mu can be done here. But the normalization of x has to be done
#    *separately* for each line, because each line has a different J and thus, in a way, different weights.
#
# 3. Define the lines and their variables

# 3. Calulate the J^scat for each line , that is a constant, incoming radiation field which is basically
#    the incoming intensity, attenuated by the optical depth in the slab. This is >>constant<<
#    for given slab model.
#
# 4. Given B, epsilon, J^scat, we can calculate the starting value for the source function S at 
#    each depth point for each line.
#    
# 5. After that - we need the calculate *total* source function and total optical depth, taking 
#    into account all the lines we are considering 
# 
# 6. Now we can solve RTE for each mu.
# 
# 7  Then we somehow iterate steps 4-6 until convergence.   

# 8. Once we have converged the source function, we can compute the emergent intensity at the surface
#    of the slab, and plot it vs wavelength.

# 9. After that we can play with different parameters, e.g. slab thickness, epsilon, B,
#    incident radiation field, and see how the emergent intensity changes. 
# 10. The endgoal is to understand how the illumination affects the line formation in the slab,
#    what impact it has on the source function. For that to be more realistic, we can make a
#    wrapper that will use all of this to fit our line to He I 1083 nm observations (Leennarts et al. 2025).
# 11. The idea is to have tau_max, epsilon, B and as free parameters to fit the observations for 
#    different heights above the limb and a given wavelength grid.


# 8. Create the class Line to handle line profile calculations, e.g. Voigt profile, Doppler width, damping parameter, etc.


# Start by defining the slab class
class Slab:
    def __init__(self, ND, tau_max, epsilon, B, H): # Still some stuff to add to the input, e.g. a = ... ,r =... and so on, think about this. 

        # Be cautious!!! r, a, epsilon, profile, etc, can be depth dependent eventually!!!
        
        self.ND = ND # Number of depth points
        self.tau_max = tau_max # Total optical depth of the slab
        self.epsilon = epsilon # Thermalization parameter at each depth
        self.B = B  # Planck function at each depth
        self.H = H  # Height above the surface
        self.tau = np.linspace(0, tau_max, ND) # This is too simple, but we will have a number of methods to calculate this
        self.S = np.zeros(ND)  # Source function initialization
        
        # Few more parameters to quantify the radiation field inside the slab
        # IM: These are now somehow obsolete, because they now belong to the line: 
        self.J_diff = np.zeros(ND) # Diffuse mean intensity
        self.L = np.zeros(ND)  # Lambda operator diagonal
        self.J_diff_lambda = 0.0 # Diffuse mean intensity per wavelength point. This a 2D array eventually
        self.J_scat = np.zeros(ND)  # Scattered mean intensity initialization
        self.r = 0.0 # line and continuum opacity ratio, eventually this shall probably be a parameter
        self.a = 0.0  # Voigt profile damping parameter

        
        # toy model absorption profile
        #self.phi = np.exp(-((np.linspace(-5, 5, 101))**2))  # Simple Gaussian profile
        #self.phi /= np.trapz(self.phi, np.linspace(-5, 5, 101))  # Normalize

        
        # store quadrature arrays (will be set in calculate_J_scat)
        self.mu_values = None
        self.mu_weights = None
        self.x_values = None
        self.x_weights = None

        self.compute_tau()  # Compute the tau grid, this function can also be used externally

        self.rel_err = np.array([])  # Relative error for convergence monitoring
        self.true_err = np.array([])  # True error for convergence monitoring

    # Now, let's create an inner class to handle line profile calculations
    class Line:
        def __init__(self, a, r, k, l_0, slab_instance, x_local=None):
            #self.d_lamb = d_lamb  # Wavelength offset
            self.a = a  # Damping parameter
            self.r = r  # Line to continuum opacity ratio
            self.k = k  # Line opacity
            self.l_0 = l_0  # Central wavelength
            self.slab_instance = slab_instance  # Reference to the parent slab instance
            self.norm = None  # Normalization factor (will be computed later)
            self.phi = None  # Profile on the line's own x grid
            self.phi_global = None  # Profile sampled on slab global x grid
            self.x_values = x_local  # local x grid (may be None until compute_profile)
            self.x_weights = None  # weights on the grid that matches x_values (used if local)
            self.J_scatter = None  # Scattered mean intensity initialization
            self.J_diff = None  # Diffuse mean intensity initialization
            self.epsilon = slab_instance.epsilon  # Thermalization parameter reference
            self.B = slab_instance.B  # Planck function reference
            
            # More parameters can be added as needed
            self.mu_values = slab_instance.mu_values
            self.mu_weights = slab_instance.mu_weights
            self.idx_range = None  # indices of global grid where this line contributes

        # Method to compute the Voigt profile on the line-local x grid (or given x)
        def compute_profile(self, x=None):
            if x is None:
                # default local grid centered on l_0 in Doppler units
                NL = getattr(self.slab_instance, "NL", 41)
                start = self.l_0 - 4
                stop = self.l_0 + 4
                x = np.linspace(start, stop, NL)
            self.x_values = x
            # make_profile expects Doppler-like offsets; keep existing behavior
            # If x is absolute-centered already, pass it through (existing code did same)
            self.slab_instance.make_profile(x, self.a, type='voigt')
            self.phi = self.slab_instance.phi.copy()
            # local normalization and local x_weights (uniform)
            dx = np.gradient(self.x_values)
            self.x_weights = dx.copy()
            norm = np.sum(self.phi * self.x_weights)
            if norm > 0:
                self.x_weights /= norm
            self.norm = norm

        # Prepare / sample this line on the slab global x grid
        def prepare_on_global(self, x_global, thresh=1e-12):
            # ensure local profile exists
            if self.x_values is None or self.phi is None:
                self.compute_profile()
            # interpolate local phi onto global grid
            self.phi_global = np.interp(x_global, self.x_values, self.phi, left=0.0, right=0.0)
            # store index range where profile is significant
            self.idx_range = np.nonzero(self.phi_global > thresh)[0]
            # set x_weights on the global grid appropriate for this line (normalized w.r.t phi_global)
            dx_global = np.gradient(x_global)
            w = dx_global.copy()
            norm = np.sum(self.phi_global * w)
            if norm > 0:
                self.x_weights = w / norm
            else:
                self.x_weights = w
            self.norm = norm

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

    def make_profile(self, x, a, type='voigt'):
        
        # Normalized Voigt profile (unit integral over x).
        # Uses scipy.special.wofz: V(x,a) = Re[w(x + i a)] / sqrt(pi).
        # Here, x is the wavelength/frequency offset in Doppler units,
        # and a is the damping parameter; x will be replaced with observed wavelength grid later.
        if type == 'voigt':
            z = x + 1j * a
            self.phi = np.real(wofz(z)) / np.sqrt(np.pi)
        elif type == 'gaussian':
            self.phi = np.exp(-x**2) / np.sqrt(np.pi)
        else:   
            raise ValueError("Unsupported profile type. Use 'voigt'.")
        
        # Ensure strict normalization on provided x grid
        #self.phi /= np.trapz(self.phi, x)

        ''' Alternative fixed profile for testing
        a, gamma = 1.0, 0.11
        a # Gaussian component HWHM
        gamma # Lorentzian component HWHM
        sigma = a / np.sqrt(2 * np.log(2))
        self.phi = np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2)))/sigma/np.sqrt(2 * np.pi)
        self.phi /= np.trapz(self.phi, x)
        '''


    def calculate_profiles_and_weights(self, NMin, NLin, verbose=False, diffuse=True, a=0.0, profile_type='voigt'):
        # This function will calculate the quadrature weights for angle and frequency
        # We will need these for two systems of reference
        x_values = np.linspace(-4, 4, NLin)  # Frequency grid
        self.make_profile(x_values, a, type=profile_type) # NL is usually 2 * range + 1
        phi = self.phi.copy()  # Copy the profile
        x_weights = np.ones_like(x_values) * (x_values[-1]-x_values[0]) / len(x_values)  # Uniform weights for simplicity
        if (verbose):
            print ("info::slab::calculate_profiles_and_weights: x_values = ", x_values)
            print (phi)
            print (x_weights)
        # Normalize the weights so the integral of the profile is 1
        norm = np.sum(phi * x_weights)
        if (verbose):
            print ("info::slab::calculate_profiles_and_weights: profile normalization before = ", norm)
        x_weights /= norm
        
        # For mu values we use Gaussian quadrature

        mu_values, mu_weights = np.polynomial.legendre.leggauss(NMin)  # 8-point Gauss-Legendre quadrature
        # We will transform according to the height H over the solar limb
        if (diffuse):
            mu_crit = 0.0  # For diffuse radiation, we integrate over all angles
        else:
            mu_crit = (1.0 - (const.R_sun.value**2.0 / (const.R_sun.value + self.H)**2.0))**0.5
        if (verbose):
            print ("info::slab::calculate_profiles_and_weights: mu_crit = ", mu_crit)
        
        # Now shift the weights and mu_values to pertain to the range [mu_crit, 1.0]
        mu_values = 0.5 * (mu_values + 1.0) * (1.0 - mu_crit) + mu_crit
        mu_weights = 0.5 * mu_weights * (1.0 - mu_crit)

        if (verbose):
            print ("info::slab::mu_values = ", mu_values)
            print ("info::slab::mu_weights = ", mu_weights)
            print ("info::slab::x_values = ", x_values)
            print ("info::slab::x_weights = ", x_weights)
            print ("info::slab::mu_norm:", np.sum(mu_weights))

        
        mu_weights /= np.sum(mu_weights)/(1.0-mu_crit)  # Normalize weights
        
        return phi, x_values, x_weights, mu_values, mu_weights
    
    # We shall define a function to calculate x grid for 2 profiles
    def calculate_profiles_and_weights_2comp(self, NMin, NLin, a1=0.1, a2=0.2, verbose=False, diffuse=True):
        # This function will calculate the quadrature weights for angle and frequency
        # We will need these for two systems of reference
        x_values = np.linspace(-4, 4, NLin)  # Frequency grid
        
        # First profile
        self.make_profile(x_values, a1, type='voigt') # NL is usually 2 * range + 1
        phi_1 = self.phi.copy()  # Copy the profile
        
        # Second profile
        self.make_profile(x_values, a2, type='voigt') # NL is usually 2 * range + 1
        phi_2 = self.phi.copy()  # Copy the profile
        
        x_weights = np.ones_like(x_values) * (x_values[-1]-x_values[0]) / len(x_values)  # Uniform weights for simplicity
        
        # Normalize the weights so the integral of the profiles is 1
        norm_1 = np.sum(phi_1 * x_weights)
        norm_2 = np.sum(phi_2 * x_weights)
        if (verbose):
            print ("info::slab::calculate_profiles_and_weights_2comp: profile 1 normalization before = ", norm_1)
            print ("info::slab::calculate_profiles_and_weights_2comp: profile 2 normalization before = ", norm_2)
        x_weights /= (norm_1 + norm_2)/2.0  # Average normalization
        
        # For mu values we use Gaussian quadrature

        mu_values, mu_weights = np.polynomial.legendre.leggauss(NMin)  # 8-point Gauss-Legendre quadrature
        # We will transform according to the height H over the solar limb
        if (diffuse):
            mu_crit = 0.0  # For diffuse radiation, we integrate over all angles
        else:
            mu_crit = (1.0 - (const.R_sun.value**2.0 / (const.R_sun.value + self.H)**2.0))**0.5
        if (verbose):
            print ("info::slab::calculate_profiles_and_weights_2comp: mu_crit = ", mu_crit)
        
        # Now shift the weights and mu_values to pertain to the range [mu_crit, 1.0]
        mu_values = 0.5 * (mu_values + 1.0) * (1.0 - mu_crit) + mu_crit
        mu_weights = 0.5 * mu_weights * (1.0 - mu_crit)

        if (verbose):
            print ("info::slab::mu_values = ", mu_values)
            print ("info::slab::mu_weights = ", mu_weights)
            print ("info::slab::x_values = ", x_values)
            print ("info::slab::x_weights = ", x_weights)
            print ("info::slab::mu_norm:", np.sum(mu_weights))

        
        mu_weights /= np.sum(mu_weights)/(1.0-mu_crit)  # Normalize weights

        return phi_1, phi_2, x_values, x_weights, mu_values, mu_weights

    def calculate_J_scat(self):
        # This function has it's own angle and frequency integration
        # So, start by creating those:
        NL = 41
        NM = 8
        phi, x_values, x_weights, mu_values, mu_weights = self.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=False)
        self.phi = phi
        self.x_values = x_values
        self.x_weights = x_weights
        self.mu_values = mu_values
        self.mu_weights = mu_weights
        self.NL = NL
        self.NM = NM

        # Now we can compute the J_inc
        J_inc = np.zeros(self.ND)
        for i in range(self.ND):
            for m in range(0, NM):
                mu = self.mu_values[m]
                w_mu = self.mu_weights[m]
                for n in range(0, NL):
                    x = self.x_values[n]
                    w_x = self.x_weights[n]
                    # Fetch the appropriate incident intensity
                    I_inc = self.get_boundary_radiation(mu)  # Simple Gaussian profile
                    # Attenuation by optical depth
                    tau_eff = (self.tau_max - self.tau[i]) / mu * self.phi[n]
                    I_attenuated = I_inc * np.exp(-tau_eff)
                    J_inc[i] += I_attenuated * self.phi[n] * w_mu * w_x / 2.0  # Divide by 2 for J    

        self.J_scat = J_inc
        del(J_inc)

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
        
    def solve_source_function(self, max_iter=1000, tol=1e-6, verbose=False):
        # As a first step, we will implement a direct matrix inversion to solve NLTE problem. No iterations needed! 
        # Just for the exercise, let's calculate new mu grid and weights here:
        NL = 41
        NM = 3
        phi, x_values, x_weights, mu_values, mu_weights = self.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=True)
        self.phi = phi
        self.x_values = x_values
        self.x_weights = x_weights
        self.mu_values = mu_values
        self.mu_weights = mu_weights
        self.NL = NL
        self.NM = NM
          
        # Here we will implement ALO method to compute the source function S
        # But first we need to calculate the full lambda operator 
        L_full = calc_lambda_full(self.tau, self.mu_values, self.mu_weights, self.phi, self.x_weights)
        A = np.eye(self.ND) - (1.0 - np.diag(self.epsilon)) * L_full
        b = self.epsilon * self.B + (1.0 - self.epsilon) * self.J_scat
        self.S = np.linalg.solve(A, b)


    def solve_source_function_ALO(self, max_iter = 1000, tol = 1e-6, verbose=False, silent=False):
        
        # Here we want to implement an interative approach to compute the source function S
        # using the ALI/ALO approach
        # Initialize S as Planck function
        self.S = self.B.copy()
        self.rel_err = np.zeros(max_iter)  # Reset relative error 
        self.true_err = np.zeros(max_iter)  # Reset true error
        SEdd = 1.0 - (1.0 - np.sqrt(self.epsilon)) * (np.exp(-np.sqrt(3.0 * self.epsilon) * self.tau)) # Eddington source function
        # Initialize weights for angle and frequency integration, same as in the function above
        NL = 17
        NM = 1
        phi, x_values, x_weights, mu_values, mu_weights = self.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=True)
        self.phi = phi
        self.x_values = x_values
        self.x_weights = x_weights
        self.mu_values = mu_values
        self.mu_weights = mu_weights
        self.NL = NL
        self.NM = NM
        #print("info::formal_solution::phi shape: ", self.phi.shape)
        #print("info::formal_solution::phi: ", self.phi)

        
          
        for iteration in range(1):
            print("info::formal_solution::iteration: ", iteration+1)
            self.J = np.zeros(self.ND)
            self.L = np.zeros(self.ND)
            self.J_diff_lambda = np.zeros((self.ND, self.NL))
            for m in range(0, self.NM):
                for l in range(0, self.NL):
                    mu = self.mu_values[m]
                    w_mu = self.mu_weights[m]
                    tau_lambda = self.tau * (self.phi[l] + self.r)
                    #print("info::formal_solution tau, l and phi: ", tau_lambda, l, self.phi[l])

                    # Outward intensity
                    I_lambda  = sc_2nd_order(tau_lambda, self.S, mu, 0.0)
                    self.J_diff_lambda[:,l] += I_lambda[0] * w_mu * 0.5
                    
                    #self.L = self.L + I_lambda[1] * self.phi[l] * self.x_weights[l] * w_mu * 0.5

                    # Inward intensity
                    I_lambda  = sc_2nd_order(tau_lambda, self.S, -mu, 0.0)
                    self.J_diff_lambda[:,l] += I_lambda[0] * w_mu * 0.5 
                    #self.L = self.L + I_lambda[1] * self.phi[l] * self.x_weights[l] * w_mu * 0.5
        
            # Sum up J over frequency
            self.J = np.sum(self.J_diff_lambda*self.phi[None,:]*self.x_weights[None,:], axis=1)
            
            print("info::formal_solution::J: ", self.J)

            # Update source function taking into account J_scat  
            if (verbose):
                print("info::formal_solution::J:", self.J)
                print("info::formal_solution::L:", self.L)
            dS = (self.epsilon * self.B + (1. - self.epsilon) * self.J + self.J_scat - self.S) / (1. - (1. - self.epsilon) * self.L)
            self.S += dS
            max_change = np.max(np.abs(dS/self.S))
            t_change = np.max(np.abs((self.S - SEdd)/SEdd))
            self.true_err[iteration] = t_change
            self.rel_err[iteration] = max_change
            # Check for convergence
            if np.max(np.abs(dS/self.S)) < tol:
                if (not silent):
                    print(f"Converged after {iteration} iterations.")
                break
        else:
            if (not silent):
                print("info::formal_solution::source function did not converge within the maximum number of iterations.")
          

    def formal_solution_given_direction(self, mu_obs, x_obs, boundary_condition, recalc_profile=False):
        # This function computes the emergent intensity at the surface of the slab
        # for a given observing angle mu_obs and observed frequency x_obs
        # for a given boundary condition (e.g., incident intensity at the bottom of the slab)

        if (recalc_profile):
            self.make_profile(x_obs, self.a, type="voigt")  # Recalculate profile for the observed frequency grid

        spectrum = np.zeros(len(x_obs))
        
        for l in range(len(x_obs)):
            # For each frequency point, compute the emergent intensity
            tau_lambda = self.tau * (self.phi[l] + self.r)
            I_emergent = sc_2nd_order(tau_lambda, self.S, mu_obs, boundary_condition)
     
            #print('!!!!!!!!', I_emergent[0].shape)
            spectrum[l] = I_emergent[0,0]  # Store the emergent intensity only, we don't need the lambda operator here
        
        return spectrum  # Return the emergent intensity spectrum only, we don't need the lambda operator here
    
    # This function is just for testing purposes
    def solve_source_function_LI_2comp(self, max_iter=1000, tol=1e-6, verbose=False):
        # Simple Lambda Iteration for two component absorption profile
        k = 8.0 # Coefficient for second component
        self.S = self.B.copy()
        S_1 = self.B.copy()
        S_2 = self.B.copy()
        S_1_hist = []
        S_2_hist = []
        NL = 41
        NM = 3
        phi_1, phi_2, x_values, x_weights, mu_values, mu_weights = self.calculate_profiles_and_weights_2comp(NM, NL, a1=0.1, a2=0.2, verbose=False, diffuse=True)
        self.phi = phi_1
        self.x_values = x_values
        self.x_weights = x_weights
        self.mu_values = mu_values
        self.mu_weights = mu_weights
        self.NL = NL
        self.NM = NM
        self.phi = phi_2
        b_per_freq = (phi_1 + 1E-5) / (phi_1 + k * phi_2 + 1E-5)  # Shape: (NL,)
        b_avg = np.sum(b_per_freq * x_weights) / np.sum(x_weights)  # Scalar: effective average weighting
        for iteration in range(max_iter):
            J_1 = np.zeros(self.ND)
            J_2 = np.zeros(self.ND)
            for m in range(0, self.NM):
                for l in range(0, self.NL):
                    mu = self.mu_values[m]
                    w_mu = self.mu_weights[m]
                    tau_lambda = self.tau * (phi_1[l] + self.r + k * phi_2[l]) 
                    #b = phi_1[l]/(phi_1[l] + k * phi_2[l])
                    
                    # S = b * S_1 + (1-b) * S_2
                    # b = phi_1 / (phi_1 + k * phi_2)
                    # S_1 = epsilon * B + (1-epsilon) * J_1
                    # S_2 = epsilon * B + (1-epsilon) * J_2
                    # J_1 = /int/int I phi_1 dphi dmu
                    # J_2 = /int/int I phi_2 dphi dmu

                    # Outward intensity
                    I_lambda = sc_2nd_order(tau_lambda, self.S, mu, 0.0)
                    J_1 += I_lambda[0] * w_mu * 0.5 * phi_1[l] * x_weights[l]
                    J_2 += I_lambda[0] * w_mu * 0.5 * phi_2[l] * x_weights[l]

                    # Inward intensity
                    I_lambda = sc_2nd_order(tau_lambda, self.S, -mu, 0.0)
                    J_1 += I_lambda[0] * w_mu * 0.5 * phi_1[l] * x_weights[l]
                    J_2 += I_lambda[0] * w_mu * 0.5 * phi_2[l] * x_weights[l]

            # Update source function taking into account J_scat  
            if (verbose):
                print("info::formal_solution::J:", J_1)
                print("info::formal_solution::J:", J_2)
            dS_1 = self.epsilon * self.B + (1. - self.epsilon) * J_1 - self.S
            dS_2 = self.epsilon * self.B + (1. - self.epsilon) * J_2 - self.S
            #dS = (phi_1/(phi_1 + k * phi_2)) * dS_1 + (k * phi_2/(phi_1 + k * phi_2)) * dS_2 + self.J_scat
            dS = b_avg * dS_1 + (1.0 - b_avg) * dS_2 + self.J_scat
            S_1 = S_1 + dS_1
            S_1_hist.append(S_1)
            S_2 = S_2 + dS_2
            S_2_hist.append(S_2)
            max_change = np.max(np.abs((dS - self.S)/self.S))
            self.S += dS
            if max_change < tol:
                if (verbose):
                    print(f"Converged after {iteration} iterations.")
                break
        else:
            if (verbose):
                print("info::formal_solution::source function did not converge within the maximum number of iterations.")

    def composite_source(self, max_iter = 1000, tol = 1e-6, verbose=False):
        self.NL = 41
        self.NM = 3
        d_lamb = 1.0
        line1 = self.Line(d_lamb=d_lamb, a=0.1, r=0.0, k=1.0, l_0=0.0, slab_instance=self, epsilon=self.epsilon)
        line2 = self.Line(d_lamb=d_lamb, a=0.2, r=0.0, k=8.0, l_0=0.0, slab_instance=self, epsilon=self.epsilon)
        line1.compute_quadrature_weights(diffuse=True, verbose=verbose)
        line2.compute_quadrature_weights(diffuse=True, verbose=verbose) 
        line1.line_J_scatter()
        line2.line_J_scatter()
        # Now solve for source functions in both lines
        line1.solve_source_func_in_line(max_iter=max_iter, tol=tol, verbose=False)
        line2.solve_source_func_in_line(max_iter=max_iter, tol=tol, verbose=False)
        # Combine the source functions
        b_per_freq = (line1.phi + 1E-5) / (line1.phi + 8.0 * line2.phi + 1E-5)  # Shape: (NL,)
        b_avg = np.sum(b_per_freq * line1.x_weights) / np.sum(line1.x_weights)  # Scalar: effective average weighting
        for iteration in range(max_iter):
            J_1 = np.zeros(self.ND)
            J_2 = np.zeros(self.ND)
            for m in range(0, self.NM):
                for l in range(0, self.NL):
                    mu = line1.mu_values[m]
                    w_mu = line1.mu_weights[m]
                    tau_lambda = self.tau * (line1.phi[l] + 8.0 * line2.phi[l]) 
                    # Outward intensity
                    I_lambda = sc_2nd_order(tau_lambda, line1.S, mu, 0.0)
                    J_1 += I_lambda[0] * w_mu * 0.5 * line1.phi[l] * line1.x_weights[l]
                    J_2 += I_lambda[0] * w_mu * 0.5 * line2.phi[l] * line2.x_weights[l]
                    # Inward intensity
                    I_lambda = sc_2nd_order(tau_lambda, line2.S, -mu, 0.0)
                    J_1 += I_lambda[0] * w_mu * 0.5 * line1.phi[l] * line1.x_weights[l]
                    J_2 += I_lambda[0] * w_mu * 0.5 * line2.phi[l] * line2.x_weights[l]
            # Update source function taking into account J_scat
            if (verbose):
                print("info::formal_solution::J:", J_1)
                print("info::formal_solution::J:", J_2)
            dS_1 = self.epsilon * self.B + (1. - self.epsilon) * J_1 + line1.J_scatter - line1.S
            dS_2 = self.epsilon * self.B + (1. - self.epsilon) * J_2 + line2.J_scatter - line2.S
            dS = b_avg * dS_1 + (1.0 - b_avg) * dS_2 + self.J_scat
            self.S += dS    
            max_change = np.max(np.abs(dS/self.S))
            if max_change < tol:
                if (verbose):
                    print(f"Converged after {iteration} iterations.")
                break
        else:
            if (verbose):
                print("info::formal_solution::source function did not converge within the maximum number of iterations.")   

# if main
if __name__ == "__main__":
    # Here we will define stuff and write proto-code to understand what is going on.

    # Define the input parameters
    ND = 81
    tau_max = 1e3
    epsilon = np.ones(ND) * 5e-3
    B = np.ones(ND) * 5.0

    # Benchmark settings
    repeats = 10

    # Helper to compute iterations done from rel_err array
    def iterations_done_from_relerr(rel):
        idx = np.flatnonzero(rel)
        return 0 if idx.size == 0 else idx[-1] + 1

    # Time direct solver (recreate slab each repeat for fair timing)
    direct_times = []
    for i in tqdm(range(repeats)):
        slab_d = Slab(ND, tau_max, epsilon, B, H=10e6)
        slab_d.calculate_J_scat()
        t0 = time.perf_counter()
        slab_d.solve_source_function()
        t1 = time.perf_counter()
        direct_times.append(t1 - t0)
    print(f"Direct solve: mean time = {np.mean(direct_times):.4f}s, std = {np.std(direct_times):.4f}s, repeats = {repeats}")
    
    # Time ALI solver (recreate slab each repeat)
    alo_times = []
    alo_iters = []
    for i in tqdm(range(repeats), desc="Timing ALI solver"):
        slab_a = Slab(ND, tau_max, epsilon, B, H=10e6)
        slab_a.calculate_J_scat()
        t0 = time.perf_counter()
        slab_a.solve_source_function_ALO(max_iter=200, tol=1e-6, verbose=False, silent=True)
        t1 = time.perf_counter()
        alo_times.append(t1 - t0)
        alo_iters.append(iterations_done_from_relerr(slab_a.rel_err))
    print(f"ALI solve: mean time = {np.mean(alo_times):.4f}s, std = {np.std(alo_times):.4f}s, mean iters = {int(np.mean(alo_iters))}")

    # One representative run (for plotting convergence / diagnostics)
    slab = Slab(ND, tau_max, epsilon, B, H=10e6)
    slab.calculate_J_scat()
    slab.solve_source_function_ALO(max_iter=500, tol=1e-8, verbose=False, silent=True)

    # Print some diagnostics
    iters_done = iterations_done_from_relerr(slab.rel_err)
    print(f"Representative ALI run: iterations = {iters_done}, last rel_err = {slab.rel_err[iters_done-1] if iters_done>0 else None}")
    print(f"Direct S (min,max) = {np.min(slab.B):.6g}, {np.max(slab.B):.6g}   ALI S (min,max) = {np.min(slab.S):.6g}, {np.max(slab.S):.6g}")

    # Plot convergence of relative error vs iteration
    rel = slab.rel_err[:iters_done]
    if rel.size > 0:
        plt.figure(figsize=(6,4))
        plt.semilogy(np.arange(1, rel.size+1), rel, marker='o')
        plt.xlabel("ALI iteration")
        plt.ylabel("relative change (max |dS/S|)")
        plt.title("ALI convergence")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("ali_convergence.png", bbox_inches='tight')
        print("Saved ali_convergence.png")
    else:
        print("No ALI iterations recorded (rel_err empty).")

    # Optionally: compare solution equality (direct vs ALI) from separate runs
    slab_direct = Slab(ND, tau_max, epsilon, B, H=10e6)
    slab_direct.calculate_J_scat()
    slab_direct.solve_source_function()
    slab_alo = Slab(ND, tau_max, epsilon, B, H=10e6)
    slab_alo.calculate_J_scat()
    slab_alo.solve_source_function_ALO(max_iter=500, tol=1e-8, verbose=False, silent=True)

    diff = np.max(np.abs(slab_direct.S - slab_alo.S))
    print(f"Max abs difference between direct and ALI S = {diff:.6e}")
    
    # Test the tau calculation
    slab = Slab(ND, tau_max, epsilon, B, H=10e6) # Note height in m
    
    
    # Next, calculate the J_scattered in the slab
    slab.calculate_J_scat()
    print ("info::main: J_scat = ", slab.J_scat)

    # Now, solve for the source function S
    #slab.solve_source_function()
    slab.solve_source_function_ALO(max_iter=200, tol=1e-6)
    S_alo = slab.S
    print ("info::main: Source function S = ", S_alo)
    
    # Now let's go for a single formal solution for given x, mu, and the boundary:
    
    x_obs = np.linspace(-10, 10, 501)  # Frequency grid
    slab.a = 0.1 # Set damping parameter
    spectrum = slab.formal_solution_given_direction(mu_obs=1.0, x_obs=x_obs, boundary_condition=1.0, recalc_profile=True)

    # plot the emergent spectrum
    plt.figure(figsize=(8,6))
    plt.plot(x_obs, spectrum, linewidth = 2, label="Emergent Spectrum")
    plt.xlabel("Frequency Offset (Doppler units)")
    plt.ylabel("Emergent Intensity")
    plt.title("Emergent Spectrum from Illuminated Finite Slab")
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"emergent_spectrum_{ND}_{tau_max}.png",bbox_inches='tight')
    plt.savefig(f"emergent_spectrum_ND_{ND}_tau_{tau_max}_eps_{epsilon[0]}.png",bbox_inches='tight')
    

    slab.solve_source_function()
    S_direct = slab.S
    print ("info::main: Source function S (ALO) = ", S_direct)

    slab.solve_source_function_LI_2comp(max_iter=200, tol=1e-6, verbose=False)
    S_2LI = slab.S

    # Plot the source function (ALO, Direct) vs tau, and J_scat vs tau

    plt.figure(figsize=(8,6))
    plt.semilogy(np.log10(slab.tau),S_alo, linestyle = "-", linewidth = 3, color = "orange", label="S ALO")
    plt.semilogy(np.log10(slab.tau),slab.J_scat, label="J_scat")
    plt.semilogy(np.log10(slab.tau),S_direct, linestyle = "--", color = "green", alpha = 1.0, label="S Direct")
    plt.semilogy(np.log10(slab.tau),S_2LI, linestyle = "-.", color = "blue", alpha = 1.0, label="S LI 2comp")
    #plt.semilogy(np.log10(slab.tau),S, linestyle = "--", color = "red", alpha = 1.0, label="Planck Function")
    #plt.plot(np.log10(slab.tau),S_direct, linestyle = "-.", color = "blue", alpha = 0.5, label="S, ALO solved")
    plt.xlabel("log10(Tau)")
    plt.ylabel("Source Function / J_scat")
    plt.title("Source Function (ALO, Direct) and J_scat vs Optical Depth")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"source_function_jscat_ND_{ND}_taum_{tau_max}_eps_{epsilon[0]}_B_{B[0]}.png",bbox_inches='tight')
    #plt.show()
    
    
    # Other variant where we use the depth index to plot all the variables
    plt.figure(figsize=(8,6))
    plt.semilogy(S_alo, linestyle = "-", linewidth = 3, color = "orange", label="S ALO")
    plt.semilogy(S_2LI, linestyle = "-.", linewidth = 3, color = "blue", label="S LI 2comp")
    plt.semilogy(slab.J_scat, label="J_scat")
    plt.semilogy(slab.B, linestyle = "--", color = "red", alpha = 1.0, label="Planck Function")
    plt.semilogy(S_direct, linestyle = "--", color = "green", alpha = 1.0, label="S Direct")
    #plt.plot(np.log10(slab.tau),S_direct, linestyle = "-.", color = "blue", alpha = 0.5, label="S, ALO solved")
    plt.xlabel("Index")
    plt.ylabel("Source Function / J_scat")
    plt.title("Source Function and J_scat vs Optical Depth")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.savefig(f"source_function_jscat_{ND}_{tau_max}_indices.png",bbox_inches='tight')
    plt.savefig(f"S_optically_thick_H={slab.H}Mm.png",bbox_inches='tight')
    

    # Now, solve for the emergent intensity given the boundary condition:
    #mu_obs = 1.0  # Observing angle cosine
    #slab.formal_solution(x, mu_obs) # Keep in mind that this function can be written in a way so it can be also used in the ALO method
    '''
    mu = np.linspace(0.0, 1.0, 100)
    I_limb = np.zeros_like(mu)
    for m in range(len(mu)):
        I_limb[m] = slab.get_boundary_radiation(mu[m])
    plt.figure(figsize=(8,6))
    plt.plot(mu, I_limb, linewidth = 2)
    plt.xlabel("mu")
    plt.ylabel("I(mu)")
    plt.title("Limb Darkening Function")
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"limb_darkening_function_{ND}_{tau_max}.png",bbox
    #print(I_limb)
    '''
    # Plot the difference between the two source functions calculated with different methods
    #plt.figure(figsize=(8,6))
    #plt.plot(S_direct - S_alo)
    #plt.xlabel("Depth Point Index")
    #plt.ylabel("Difference in Source Function")
    #plt.title("Difference between Direct and ALO Source Functions") 
    #plt.show()