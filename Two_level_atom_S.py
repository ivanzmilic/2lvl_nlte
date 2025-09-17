from scipy.special import wofz
import numpy as np
import matplotlib.pyplot as plt
from rtfunctions import one_full_fs, sc_2nd_order, calc_lambda_full, calc_lambda_monoc

def solve_2level_nlte(ND, NM, NL, B, profile_type, lratio = 1E3, slab = False):
    # ND - number of points for depth/optical depth grid
    # NM - number of points for angle grid
    # NL - number of points for wavelength/frequency grid
    # B - Planck's function
    # profile_type - spectral line profile (Dopple, Voigt or Lorentz)
    # lratio (default = 1E3 for us) is the line and continuum opacity 
    # slab - true if the user studies slab of finite optical depth, false if user is interested in semi-infinite atmopshere; default is False 
    
    # Optical depth grid
    if (slab == False):
        logtau = np.linspace(-5, 10, ND)
        tau = logtau**10
        up_bound = 0.0
        low_bound = 1.0
    else:
        T = float(input("Please enter the value for total optical thickness of slab: "))
        ll = np.log10(T)
        logtau = np.linspace(-ll, ll, ND)
        tau = 10**logtau
        up_bound = 0.0
        low_bound = 0.0

    # Planck's function
    B = np.zeros(ND)
    B[:] = 1 # constant with depth (simplified case)
    #print(B)
    
    # Photon probablity destruction
    epsilon = np.zeros(ND)
    epsilon[:] = float(input("Please enter the value for photon destruction probability: ")) # change to suit your needs

    # Reduced wavlength in Doppler widths
    x = np.linspace(-5, 5, NL)
    #print(x)
    
    # Number of iterations
    #N_iter = float(input("Please enter the number of iterations: "))

    if (profile_type == 1):
        # Doppler profile
        profile = 1/np.sqrt(np.pi) * np.exp(-(x**2))
    elif profile_type == 2:
        # Voigt profile
        alpha = float(input("Please enter the value for alpha: "))
        gamma = float(input("Please enter the value for gamma: "))
        sigma = alpha / np.sqrt(2 * np.log(2))
        profile = np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2)))/sigma/np.sqrt(2 * np.pi)
    elif profile_type == 3:
        # Lorentz profile
        profile = 1/np.pi * (1 + x**2)
    else:
        print("Profile type must be 1, 2 or 3")

    # Quadrature
    wx = np.zeros(NL)
    wx[0] = (x[1] - x[0]) * 0.5
    wx[-1] = (x[-1] - x[-2]) * 0.5
    wx[1:-1] = (x[2:NL] - x[0:-2]) * 0.5
    norm = (np.sum(profile*wx))
    wx = wx/norm
    
    # Angle integration:
    mu = ([1./np.sqrt(3.0)])
    wmu = [1.0]
    mu = np.cos([0.4793425352,1.0471975512,1.4578547042])
    wmu = [.2777777778,0.4444444444,0.2777777778]
    NM = mu.shape[0]
    mu = np.asarray(mu)
    wmu = np.asarray(wmu)
    #print(wmu)
    #print(profile)
    #print(mu)
    # Initialize one figure for plotting
    fig = plt.figure(constrained_layout = True, figsize = (13, 7))

    # Solve radiative transfer equation (first by using Ivan's functions)
    S = np.copy(B) # the source function 
    #print(S)
    # We will iterate 'till the convergence
    for j in range(0, 200):
        J = np.zeros(ND) # initialize the scattering integral
        L = np.zeros(ND) # initialize the Lambda operator
        # For each direction and wavelength, calculate monochromatic intensity and add its contribution to the scattering integral
        for m in range(0, NM):
            for l in range(0, NL):

                # outward

                I_Lambda = sc_2nd_order(tau * profile[l] * lratio, S, mu[m], low_bound)

                J = J + I_Lambda[0] * profile[l] * wx[l] * wmu[m] * 0.5

                L = L + I_Lambda[1] * profile[l] * wx[l] * wmu[m] * 0.5

                # inward

                I_Lambda = sc_2nd_order(tau * profile[l] * lratio, S, -mu[m], up_bound)

                J = J + I_Lambda[0] * profile[l] * wx[l] * wmu[m] * 0.5

                L = L + I_Lambda[1] * profile[l] * wx[l] * wmu[m] * 0.5

        # Correct the source function using the local ALI approach
        dS = (epsilon * B + (1. - epsilon) * J - S)/(1. - (1. - epsilon) * L)

        # Check for change
        max_change = np.max(np.abs(dS/S))

        S += dS
        #print(S)
        plt.semilogy(logtau, S, '-k', alpha = 0.20)
        if(max_change < 1E-4):
            break
        fin = S    
    plt.xlabel("$\\log\\tau$ in the line")
    plt.ylabel("$\\log$S")
    plt.show()
    I_out = one_full_fs(tau * lratio, fin, 1.0, profile, fin[-1])
    #plt.tight_layout()
    return np.stack([fin, I_out])





# A basic example
S = solve_2level_nlte(91, 3, 21, 1.0, 1, lratio = 1E3, slab = True)[0]
I = solve_2level_nlte(91, 3, 21, 1.0, 1, lratio = 1E3, slab = True)[1]