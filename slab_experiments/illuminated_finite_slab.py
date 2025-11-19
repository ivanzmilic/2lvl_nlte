import numpy as np 
import matplotlib.pyplot as plt
import astropy.units as units
import astropy.constants as const

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
    def __init__(self, ND, tau_max, epsilon, B, H):
        
        self.ND = ND # Number of depth points
        self.tau_max = tau_max # Total optical depth of the slab
        self.epsilon = epsilon # Thermalization parameter at each depth
        self.B = B  # Planck function at each depth
        self.H = H  # Height above the surface
        self.tau = np.linspace(0, tau_max, ND) # This is too simple, but we will have a number of methods to calculate this
        self.S = np.zeros(ND)  # Source function initialization
        self.J_scat = np.zeros(ND)  # Scattered mean intensity initialization

        # toy model absorption profile
        self.phi = np.exp(-((np.linspace(-5, 5, 101))**2))  # Simple Gaussian profile
        self.phi /= np.trapz(self.phi, np.linspace(-5, 5, 101))  # Normalize

        # eventually this shall probably be a parameter
        self.r = 1 # line and continuum opacity ratio

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

    def calculate_weights(self, NMin, NLin, verbose=False):
        # This function will calculate the quadrature weights for angle and frequency
        # We will need these for two systems of reference
        self.NL = NLin
        x_values = np.linspace(-5, 5, self.NL)  # Frequency grid
        x_weights = np.ones_like(x_values) / len(x_values)  # Uniform weights for simplicity

        # For mu values we use Gaussian quadrature
        self.NM = NMin
        mu_values, mu_weights = np.polynomial.legendre.leggauss(self.NM)  # 8-point Gauss-Legendre quadrature
        # We will transform according to the height H over the solar limb
        mu_crit = (1.0 - (const.R_sun.value**2.0 / (const.R_sun.value + self.H)**2.0))**0.5
        if (verbose):
            print ("info::slab::calculate_J_scat: mu_crit = ", mu_crit)
        
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
        NL = 101
        NM = 8
        self.calculate_weights(NM, NL, verbose=False)

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
            I_0 = 0.4 + 0.6 * mu_0
            return I_0
        
    def solve_source_function(self, max_iter=1000, tol=1e-6, verbose=False):
        # As a first step, we will implement a direct matrix inversion to solve NLTE problem. No iterations needed! 
        # Just for the exercise, let's calculate new mu grid and weights here:
        NL = 101
        NM = 3
        self.calculate_weights(NM, NL, verbose=False)
          
        # Here we will implement ALO method to compute the source function S
        # But first we need to calculate the full lambda operator 
        L_full = calc_lambda_full(self.tau, self.mu_values, self.mu_weights, self.phi, self.x_weights)
        A = np.eye(self.ND) - (1.0 - np.diag(self.epsilon)) * L_full
        b = self.epsilon * self.B + (1.0 - self.epsilon) * self.J_scat
        self.S = np.linalg.solve(A, b)


    def formal_solution(self, max_iter = 1000, tol = 1e-6):
        # Here we want to implement the formal solution to compute the source function S
        # using the ALI approach
        # Initialize S as Planck function
        self.S = self.B.copy()
        for iteration in range(max_iter):
            J = np.zeros(self.ND)
            L = np.zeros(self.ND)
            for m in range(0, self.NM):
                for l in range(0, self.NL):
                    mu = self.mu_values[m]
                    w_mu = self.mu_weights[m]

                    # Outward intensity
                    I_lambda  = sc_2nd_order(self.tau * self.phi[l] * self.r, self.S, mu[m], 1.0)
                    J = J + I_lambda[0] * self.phi[l] * self.x_weights[l] * w_mu[m] * 0.5
                    L = L + I_lambda[1] * self.phi[l] * self.x_weights[l] * w_mu[m] * 0.5

                    # Inward intensity
                    I_lambda  = sc_2nd_order(self.tau * self.phi[l] * self.r, self.S, -mu[m], 0.0)
                    J = J + I_lambda[0] * self.phi[l] * self.x_weights[l] * w_mu[m] * 0.5
                    L = L + I_lambda[1] * self.phi[l] * self.x_weights[l] * w_mu[m] * 0.5
            # Update source function
            dS = (self.epsilon * self.B + (1. - self.epsilon) * J - self.S) / (1. - (1. - self.epsilon) * L)
            # Check for convergence
            if np.max(np.abs(dS/self.S)) < tol:
                print(f"Converged after {iteration} iterations.")
                break
        else:
            print("info::formal_solution::source function did not converge within the maximum number of iterations.")

# if main
if __name__ == "__main__":
    # Here we will define stuff and write proto-code to understand what is going on.

    # Define the input parameters
    ND = 101
    tau_max = 1000.0
    epsilon = np.ones(ND) * 1e-6
    B = np.ones(ND) * 10000.0

    # Test the tau calculation
    slab = Slab(ND, tau_max, epsilon, B, H=80e6) # Height of 80 Mm
    
    
    # Next, calculate the J_scattered in the slab
    slab.calculate_J_scat()
    #print ("info::main: J_scat = ", slab.J_scat)

    # Now, solve for the source function S
    slab.solve_source_function()
    print ("info::main: Source function S = ", slab.S)
    
    # Plot the source function vs tau, and J_scat vs tau
    plt.figure(figsize=(8,6))
    plt.plot(np.log10(slab.tau),slab.S)
    plt.plot(np.log10(slab.tau),slab.J_scat)
    plt.xlabel("log10(Tau)")
    plt.ylabel("Source Function / J_scat")
    plt.title("Source Function and J_scat vs Optical Depth")
    plt.legend(["Source Function S","J_scat"])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"source_function_jscat_{ND}_{tau_max}.png",bbox_inches='tight')
    #plt.show()

    # Now, solve for the emergent intensity given the boundary condition:
    #mu_obs = 1.0  # Observing angle cosine
    #slab.formal_solution(x, mu_obs) # Keep in mind that this function can be written in a way so it can be also used in the ALO method