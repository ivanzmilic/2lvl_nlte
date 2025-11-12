import numpy as np 
import matplotlib.pyplot as plt
import astropy.units as units
import astropy.constants as const

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
        
    def calculate_J_scat(self):
        # This function has it's own angle and frequency integration
        # So, start by creating those:
        x_values = np.linspace(-5, 5, 101)  # Frequency grid
        x_weights = np.ones_like(x_values) / len(x_values)  # Uniform weights for simplicity

        # For mu values we use Gaussian quadrature
        mu_values, mu_weights = np.polynomial.legendre.leggauss(8)  # 8-point Gauss-Legendre quadrature
        # We will transform according to the height H over the solar limb
        mu_crit = (1.0 - (const.R_sun.value**2.0 / (const.R_sun.value + self.H)**2.0))**0.5
        print ("info::slab::calculate_J_scat: mu_crit = ", mu_crit)
        
        # Now shift the weights and mu_values to pertain to the range [mu_crit, 1.0]
        mu_values = 0.5 * (mu_values + 1.0) * (1.0 - mu_crit) + mu_crit
        mu_weights = 0.5 * mu_weights * (1.0 - mu_crit)
        print ("info::slab::calculate_J_scat: mu_values = ", mu_values)
        print ("info::slab::calculate_J_scat: mu_weights = ", mu_weights)
        # Note that the mu_weights will sum to (1-mu_crit), which is what we want.
        # And additionally, later we will have to multiply by 2 to get appropriate J.

        # Now we can compute the J_inc
        #J_inc = np.zeros(self.ND)
        #for i in range(self.ND):
        #    J_inc[i] = np.sum(self.B[i] * x_weights)

        #return J_inc

# if main
if __name__ == "__main__":
    # Here we will define stuff and write proto-code to understand what is going on.

    # Define the input parameters
    ND = 101
    tau_max = 10.0
    epsilon = np.ones(ND) * 1e-6
    B = np.ones(ND) * 1.0

    # Test the tau calculation
    slab = Slab(ND, tau_max, epsilon, B, H=80e6) # Height of 80 Mm
    slab.compute_tau()
    
    # Next, calculate the J_scattered in the slab
    slab.calculate_J_scat()