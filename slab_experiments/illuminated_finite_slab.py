import numpy as np 
import matplotlib.pyplot as plt
import astropy.units as units
import astropy.constants as const
from scipy.special import wofz
from rtfunctions import one_full_fs, sc_2nd_order, calc_lambda_full, calc_lambda_monoc

# Some basic description, so we know what we are doing: 
# This script solves the NLTE line formation for a 2-level atom in an illuminated finite slab.
# The slab is illuminated from one side by a radiation field with a given angular 
# and spectral distribution. The goal is to compute the emergent intensity and the source function 
# within the slab, taking into account both scattering and thermal emission.

# The flow is something like this: 
# 1. Define epsilon, B, parameters necessary for the calculation of line profile 
#    (for example, we might not want to have constant line profile, but something that can 
#    vary with location, and perhaps be shifted due to velocity fields).
# 2. Calulate the J^scat, that is a constant, incoming radiation field which is basically
#    the incoming intensity, attenuated by the optical depth in the slab. This is >>constant<<
#    for given slab model.
# 3. Given B, epsilon, J^scat, we can calculate the source function S at each depth point.
#    For this we use the standard formal solution that we used since the start 
#    and we use the good old ALO method to accelerate convergence.
# 4. Once we have the source function, we can compute the emergent intensity at the surface
#    of the slab, and plot it vs wavelength.

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
        self.J_scat = np.zeros(ND)  # Scattered mean intensity initialization
        self.r = 0 # line and continuum opacity ratio, eventually this shall probably be a parameter
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

    def compute_tau(self):
        # As the most robust method we will use log-spaced grid on both sides.
        # First we check the total ND, and make sure it's odd
        if self.ND % 2 == 0:
            self.ND += 1  # Make it odd
        
        # Create log-spaced grid for first half
        half_ND = self.ND // 2 + 1
        tau_first_half = np.logspace(-4, np.log10(self.tau_max / 2.0), half_ND)
        # Create log-spaced grid for second half
        # A bit of cheating because we want to get exact total tau_max
        tau_second_half = self.tau_max + tau_first_half[0] - tau_first_half[::-1]
        # Put these two together
        self.tau = np.concatenate((tau_first_half, tau_second_half[1:]))

    def voigt_profile(self, x, a):
        
        # Normalized Voigt profile (unit integral over x).
        # Uses scipy.special.wofz: V(x,a) = Re[w(x + i a)] / sqrt(pi).
        # Here, x is the wavelength/frequency offset in Doppler units,
        # and a is the damping parameter; x will be replaced with observed wavelength grid later.
        z = x + 1j * a
        self.phi = np.real(wofz(z)) / np.sqrt(np.pi)
        # Ensure strict normalization on provided x grid
        self.phi /= np.trapz(self.phi, x)


    def calculate_profiles_and_weights(self, NMin, NLin, verbose=False, diffuse=True):
        # This function will calculate the quadrature weights for angle and frequency
        # We will need these for two systems of reference
        self.NL = NLin
        x_values = np.linspace(-10, 10, self.NL)  # Frequency grid
        self.voigt_profile(x_values, 0.0) # NL is usually 2 * range + 1
        x_weights = np.ones_like(x_values) / len(x_values)  # Uniform weights for simplicity
        # Normalize the weights so the integral of the profile is 1
        norm = np.trapz(self.phi, x_values)
        if (verbose):
            print ("info::slab::calculate_profiles_and_weights: profile normalization before = ", norm)
        x_weights /= norm
        
        # For mu values we use Gaussian quadrature

        self.NM = NMin
        mu_values, mu_weights = np.polynomial.legendre.leggauss(self.NM)  # 8-point Gauss-Legendre quadrature
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

        self.mu_values = mu_values
        self.mu_weights = mu_weights
        self.x_values = x_values
        self.x_weights = x_weights
        

    def calculate_J_scat(self):
        # This function has it's own angle and frequency integration
        # So, start by creating those:
        NL = 41
        NM = 8
        self.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=False)

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
                    I_inc = self.get_boundary_radiation(mu, x)  # Simple Gaussian profile
                    # Attenuation by optical depth
                    tau_eff = (self.tau_max - self.tau[i]) / mu * self.phi[n]
                    I_attenuated = I_inc * np.exp(-tau_eff)
                    J_inc[i] += I_attenuated * w_mu * w_x / 2.0  # Divide by 2 for J    

        self.J_scat = J_inc
        del(J_inc)

    def get_boundary_radiation(self, mu, x):
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
            # I_0 = 1 - 0.5311 * (1 - mu_0**0.5) + 0.0545 * (1 - mu_0) - 0.7301 * (1 - mu_0**1.5) + 0.4053 * (1 - mu_0**2)
            I_0 = 0.4 + 0.6 * mu_0
            return I_0
        
    def solve_source_function(self, max_iter=1000, tol=1e-6, verbose=False):
        # As a first step, we will implement a direct matrix inversion to solve NLTE problem. No iterations needed! 
        # Just for the exercise, let's calculate new mu grid and weights here:
        NL = 41
        NM = 3
        self.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=True)
          
        # Here we will implement ALO method to compute the source function S
        # But first we need to calculate the full lambda operator 
        L_full = calc_lambda_full(self.tau, self.mu_values, self.mu_weights, self.phi, self.x_weights)
        A = np.eye(self.ND) - (1.0 - np.diag(self.epsilon)) * L_full
        b = self.epsilon * self.B + (1.0 - self.epsilon) * self.J_scat
        self.S = np.linalg.solve(A, b)


    def solve_source_function_ALO(self, max_iter = 1000, tol = 1e-6):
        
        # Here we want to implement an interative approach to compute the source function S
        # using the ALI/ALO approach
        # Initialize S as Planck function
        self.S = self.B.copy()

        # Initialize weights for angle and frequency integration, same as in the function above
        NL = 41
        NM = 3
        self.calculate_profiles_and_weights(NM, NL, verbose=False, diffuse=True)
          
        for iteration in range(max_iter):
            
            J = np.zeros(self.ND)
            L = np.zeros(self.ND)
            for m in range(0, self.NM):
                for l in range(0, self.NL):
                    mu = self.mu_values[m]
                    w_mu = self.mu_weights[m]
                    tau_lambda = self.tau * (self.phi[l] + self.r)
                    #print(tau_lambda.shape, self.S.shape, mu)
                    

                    # Outward intensity
                    I_lambda  = sc_2nd_order(tau_lambda, self.S, mu, 0.0)
                    J = J + I_lambda[0] * self.phi[l] * self.x_weights[l] * w_mu * 0.5
                    L = L + I_lambda[1] * self.phi[l] * self.x_weights[l] * w_mu * 0.5

                    # Inward intensity
                    I_lambda  = sc_2nd_order(tau_lambda, self.S, -mu, 0.0)
                    J = J + I_lambda[0] * self.phi[l] * self.x_weights[l] * w_mu * 0.5
                    L = L + I_lambda[1] * self.phi[l] * self.x_weights[l] * w_mu * 0.5
            # Update source function taking into account J_scat  
            print(J)
            dS = (self.epsilon * self.B + (1. - self.epsilon) * J + self.J_scat - self.S) / (1. - (1. - self.epsilon) * L)
            self.S += dS
            # Check for convergence
            if np.max(np.abs(dS/self.S)) < tol:
                print(f"Converged after {iteration} iterations.")
                break
        else:
            print("info::formal_solution::source function did not converge within the maximum number of iterations.")

    def formal_solution_given_direction(self, mu_obs, x_obs, boundary_condition, recalc_profile=False):
        # This function computes the emergent intensity at the surface of the slab
        # for a given observing angle mu_obs and observed frequency x_obs
        # for a given boundary condition (e.g., incident intensity at the bottom of the slab)

        if (recalc_profile):
            self.voigt_profile(x_obs, self.a)

        spectrum = np.zeros(len(x_obs))
        
        for l in range(len(x_obs)):
            # For each frequency point, compute the emergent intensity
            tau_lambda = self.tau * (self.phi[l] + self.r)
            I_emergent = sc_2nd_order(tau_lambda, self.S, mu_obs, boundary_condition)
     
            #print('!!!!!!!!', I_emergent[0].shape)
            spectrum[l] = I_emergent[0,0]  # Store the emergent intensity only, we don't need the lambda operator here
        
        return spectrum  # Return the emergent intensity spectrum only, we don't need the lambda operator here
        
# if main
if __name__ == "__main__":
    # Here we will define stuff and write proto-code to understand what is going on.

    # Define the input parameters
    ND = 41
    tau_max = 10000.0
    epsilon = np.ones(ND) * 1e-2
    B = np.ones(ND) * 1.0

    # Test the tau calculation
    slab = Slab(ND, tau_max, epsilon, B, H=10e6) # Note height in m
    
    
    # Next, calculate the J_scattered in the slab
    #slab.calculate_J_scat()
    #print ("info::main: J_scat = ", slab.J_scat)

    # Now, solve for the source function S
    #slab.solve_source_function()
    slab.solve_source_function_ALO()
    S_direct = slab.S
    print ("info::main: Source function S = ", S_direct)

    # Now let's go for a single formal solution for given x, mu, and the boundary:

    x_obs = np.linspace(-10, 10, 501)  # Frequency grid
    slab.a = 0.00  # Set damping parameter
    spectrum = slab.formal_solution_given_direction(mu_obs=1.0, x_obs=x_obs, boundary_condition=0.0, recalc_profile=True)

    # plot the emergent spectrum
    plt.figure(figsize=(8,6))
    plt.plot(x_obs, spectrum, linewidth = 2, label="Emergent Spectrum")
    plt.xlabel("Frequency Offset (Doppler units)")
    plt.ylabel("Emergent Intensity")
    plt.title("Emergent Spectrum from Illuminated Finite Slab")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"emergent_spectrum_{ND}_{tau_max}.png",bbox_inches='tight')
    
    '''
    slab.solve_source_function_ALO()
    S_alo = slab.S
    print ("info::main: Source function S (ALO) = ", S_alo)

    # Plot the source function vs tau, and J_scat vs tau
    plt.figure(figsize=(8,6))
    plt.plot(np.log10(slab.tau),S_direct, linestyle = "-", linewidth = 3, color = "orange", label="S, directly solved")
    plt.plot(np.log10(slab.tau),slab.J_scat, label="J_scat")
    plt.plot(np.log10(slab.tau),S_alo, linestyle = "-.", color = "blue", alpha = 0.5, label="S, ALO solved")
    plt.xlabel("log10(Tau)")
    plt.ylabel("Source Function / J_scat")
    plt.title("Source Function and J_scat vs Optical Depth")
    plt.legend(["Source Function S","J_scat","Source Function S (ALO)"])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"source_function_jscat_{ND}_{tau_max}.png",bbox_inches='tight')
    #plt.show()

    # Now, solve for the emergent intensity given the boundary condition:
    #mu_obs = 1.0  # Observing angle cosine
    #slab.formal_solution(x, mu_obs) # Keep in mind that this function can be written in a way so it can be also used in the ALO method
    '''