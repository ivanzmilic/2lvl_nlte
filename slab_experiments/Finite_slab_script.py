# This code solves 2 level atom for non-LTE case in finite, free-standing (non-)illuminated slabs.
# The idea is to expand upon examples given in Mihalas (1978) and references therin;
# this means that we shall first reproduce the examples in questions and then 
# strive to correctly solve illuminated case.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from rtfunctions import one_full_fs, sc_2nd_order, calc_lambda_full, calc_lambda_monoc
from scipy.integrate import quad as q

# We shall define functions, i.e., for solving (solver), for quadrature and so on...

# Define "constants"
R_sun = 696000 # km
H = 80000 # km above the Sun's surface
tht_crit = R_sun/(R_sun + H)
mu_crit = np.cos(tht_crit)

def J_photosphere(tau, quad):
    ND = len(tau)
    tau_LOS = np.zeros([len(quad[3]), len(quad[1])])
    for m in range(0, len(quad[3])):
        for l in range(0, len(quad[1])):
            tau_LOS[m][l] = (tau[-1] - tau[l]) / quad[3][m]
            #print(tau_LOS[m][l])
    S = np.ones(ND)
    '''
    for j in range(0, 500):
        J_const = np.zeros(ND)
        L = np.zeros(ND)
        for m in range(0, len(quad[3])):
            for l in range(0, len(quad[1])):

                I = sc_2nd_order(tau_los, S, quad[3], 0.0)

                J_const = J_const + I[0] * quad[0][l] * quad[2][l] * quad[4][m] * 0.5

                L = L + I[1] * quad[0][l] * quad[2][l] * quad[4][m] * 0.5
    '''
    #I_ph = one_full_fs(tau_los, S, quad[3], quad[0], 0.0)
    #J_const, err = q(I_ph, mu_crit, 1)
    return tau_LOS

def two_level_nlte(tau, quad, B, eps, lower_bound, upper_bound, r, niter):

    # This is the solver that computes S using ALI scheme

    # tau - optical depth grid
    # mu - angle grid
    # x - reduced wavelength grid
    # B - Planck's function
    # eps = epsilon - photon destruction probability
    # lower_bound - intensity at lower bound
    # upper_bound - intensity at upper bound
    # r; \tau_\lambda = \tau_0 * (\varphi_\lambda + r)

    ND = len(tau) # number of optical depth points
    logtau = np.log10(tau)

    # Planck's function
    B_arr = np.zeros(ND)
    B_arr[:] = B

    # Photon destruction probability
    epsilon = np.zeros(ND)
    epsilon[:] = eps

    # Number of points for angle grid
    NM = len(quad[3])
    mu = np.asarray(quad[3])
    wmu = np.asarray(quad[4])


    # Number of points for reduced wavelength grid
    NL = len(quad[1])
    profile = np.asarray(quad[0])
    wx = np.asarray(quad[2])

     # One figure for plotting the source function
    fig = plt.figure(constrained_layout = True, figsize = (13, 7))

    if(niter < 500):
        # Relative error
        rel_err = np.zeros((niter))
        res = [] # in case we want this function to return [S, Lamba_operator]
        
        # Solve radiative transfer equation (first by using Ivan's functions)
        S = np.copy(B_arr) # the source function 
        # We will iterate 'till the convergence
        for j in range(0, niter):
            J = np.zeros(ND) # initialize the scattering integral
            L = np.zeros(ND) # initialize the Lambda operator
            # For each direction and wavelength, calculate monochromatic intensity and add its contribution to the scattering integral
            for m in range(0, NM):
                for l in range(0, NL):

                    # outward

                    I_Lambda = sc_2nd_order(tau * profile[l] * r, S, mu[m], lower_bound)

                    J = J + I_Lambda[0] * profile[l] * wx[l] * wmu[m] * 0.5

                    L = L + I_Lambda[1] * profile[l] * wx[l] * wmu[m] * 0.5

                    # inward

                    I_Lambda = sc_2nd_order(tau * profile[l] * r, S, -mu[m], upper_bound)

                    J = J + I_Lambda[0] * profile[l] * wx[l] * wmu[m] * 0.5

                    L = L + I_Lambda[1] * profile[l] * wx[l] * wmu[m] * 0.5

            # Correct the source function using the local ALI approach
            dS = (epsilon * B_arr + (1. - epsilon) * J - S) / (1. - (1. - epsilon) * L)

            # Check for change
            max_change = np.max(np.abs(dS/S))
            rel_err[j] = max_change
            #print(rel_err[j])

            S += dS
            plt.semilogy(logtau, S, '-k', alpha = 0.20)
            if(max_change < 1E-4):
                break
            fin = S    
        plt.xlabel("$\\log\\tau$ in the line")
        plt.ylabel("$\\log$S")
        plt.show()
    else:
        # Lambda iteration
        # Relative error
        rel_err = np.zeros((niter))
        res = [] # in case we want this function to return [S, Lamba_operator]
        
        # Solve radiative transfer equation (first by using Ivan's functions)
        S = np.copy(B_arr) # the source function 
        for iter in range(0, niter):
        # initialize the scattering integral
            J = np.zeros(ND)
            for m in range(0, NM):
                for l in range(0, NL):
                    #outward
                    I = sc_2nd_order(tau * profile[l] * r, S, mu[m], lower_bound)[0]
                    J += I * profile[l] * wx[l] * wmu[m] * 0.5
                    #inward
                    I = sc_2nd_order(tau * profile[l] * r, S, -mu[m], upper_bound)[0]
                    J += I * profile[l] * wx[l] * wmu[m] * 0.5
            
            dS = eps * B + (1.-eps) * J - S
            max_change = np.amax(np.abs(dS/S))
            #print (max_change)
            S += dS
            plt.semilogy(logtau-3, S,'-k', alpha = 0.2)
            if(max_change < 1E-4):
                    break
            fin = S    
        plt.xlabel("$\\log\\tau$ in the line")
        plt.ylabel("Source function")
        plt.show()
    #LL = calc_lambda_full(tau * lratio, mu, wmu, profile, wx)
    # res.append(fin)
    # res.append(LL)
    # return res
    return fin

    return

def quadrature(NL, profile_type):
    # Here we compute quadrature for numerical calculation

    # NL - number of wavelength points

    # Reduced wavlength in Doppler widths
    x = np.linspace(-6, 6, NL)

    # Profile type: Doppler, Voight or Lorentz
    if (profile_type == 1):
        # Doppler profile
        profile = 1/np.sqrt(np.pi) * np.exp(-(x**2))
    elif profile_type == 2:
        # Voigt profile
        alpha = float(input("Please enter the value for alpha: "))
        gamma = float(input("Please enter the value for gamma: "))
        sigma = alpha / np.sqrt(2 * np.log(2))
        # sigma = 1
        profile = np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2)))/sigma/np.sqrt(2 * np.pi)
    elif profile_type == 3:
        # Lorentz profile
        profile = 1/np.pi * (1 + x**2)
    else:
        print("Profile type must be 1, 2 or 3")
    
    # Weights for wavelengths
    
    wx = np.zeros(NL)
    wx[0] = (x[1] - x[0]) * 0.5
    wx[-1] = (x[-1] - x[-2]) * 0.5
    wx[1:-1] = (x[2:NL] - x[0:-2]) * 0.5
    norm = (np.sum(profile*wx))
    wx = wx/norm
    
    # Angle integration:
    
    mu=([1./np.sqrt(3.0)])
    wmu=[1.0]

    # Third approximation
    #mu=np.cos([0.4793425352,1.0471975512,1.4578547042])
    #wmu=[.2777777778,0.4444444444,0.2777777778]
    
    #Fourth approximation
    mu = np.array([0.06943184, 0.33000948, 0.66999052, 0.93056816])
    wmu = [0.173927419815906, 0.326072580184089, 0.326072580184104, 0.173927419815900]
    
    NM = mu.shape[0]
    mu = np.asarray(mu)
    wmu = np.asarray(wmu)

    return [profile, x, wx, mu, wmu]

# Let's define a function that defines the finite slab in terms of optical depth 
# and return the log_tau grid

# We want a slab with given optical thickness taumax, and some minimum thickne at the surface. 
# And we want given number of points per decade. 
# AND we want the slab to be symmetric around the middle, meaning very fine separation at the edges and coarse 
# in the middle.

def tau_grid(taumin, taumax, np_per_dec):
    # How many decades in total?
    n_decades = np.log10(taumax / taumin)
    print(n_decades)

    # How many decades until the middle of the slab?
    n_dec_mid = np.log10(taumax / taumin / 2)
    print(n_dec_mid)

    # Total number of points is then: 
    ND = int (np_per_dec * n_dec_mid) + 1
    print(ND)

    # Now, the step is, actually: 
    logstep = np.log10(taumax / taumin / 2) / (ND - 1)
    print(logstep)

    # Make log grid for a half of the slab:
    logtau_half = np.linspace(np.log10(taumin), np.log10(taumax/2), ND)

    tau_half = 10**logtau_half

    #print(tau_half)

    tau_second_half = taumax - tau_half[::-1]

    tau_full = np.concatenate((tau_half, tau_second_half[1:]))

    print(tau_full)

    return tau_full

# Let us recreate values from Mihalas (1978)
tau = tau_grid(1E-4, 1E4, 12)
quad = quadrature(121, 1)


S_d_e2_T4 = two_level_nlte(tau, quad, 1.0, 1E-6, 0.0, 0.0, 1, 400)

J_ph = J_photosphere(tau, quad)
print(J_ph)