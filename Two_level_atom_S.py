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
    
    npdec = 22 # p in Rybicki & Hummer (1991), number of points per decade 

    # Optical depth grid
    if (slab == False):
        logtau = np.linspace(-5, 10, ND)
        tau = logtau**10
        up_bound = 0.0
        low_bound = 1.0
    else:
        dtau1 = 1E-2
        taumax = 1E4
        fac = 10.**(1./float(npdec))
        dtau = dtau1  
        t = 0.
        k = 1
        while (t < taumax):
            k = k + 1
            t = t + dtau
            dtau = fac * dtau
        npts = k
        z = np.zeros((npts))
        z[0] = 0.
        z[1] = dtau1
        dtau = dtau1 * fac
        for k in np.arange(2, npts):
            z[k] = z[k-1] + dtau
            dtau = fac * dtau
        tau = z
        ND = npts
        print(tau)
        '''
        T = float(input("Please enter the value for total optical thickness of slab: "))
        ll = np.log10(T)
        logtau_deep = np.linspace(-ll, 0 , ND)
        logtau_shallow = np.linspace(0, ll, ND)
        logtau = np.concatenate((logtau_deep, logtau_shallow))
        tau = 10**logtau
        '''
        up_bound = 0.0
        low_bound = 1.0

        
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
    norm = (np.sum(profile * wx))
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

    # Relative and true error 
    SEdd = 1.0 - (1.0 - np.sqrt(epsilon)) * (np.exp(-np.sqrt(3.0 * epsilon) * tau)) # Eddington source function

    true_err = np.zeros((200))
    rel_err = np.zeros((200))


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
        rel_err[j] = max_change

        true_error = np.max(np.abs(S - SEdd)/SEdd)
        true_err[j] = true_error

        S += dS
        #print(S)
        plt.semilogy(np.log10(tau), S, '-k', alpha = 0.20)
        plt.semilogy(np.log10(tau), SEdd, '--', color = "red", linewidth = 2)
        if(max_change < 1E-6):
            break
        fin = S    
    plt.xlabel("$\\log\\tau$ in the line")
    plt.ylabel("$\\log$S")
    plt.show()
    #tau_LL = tau**10
    I_out = one_full_fs(tau * lratio, fin, 1.0, profile, fin[-1])
    LL = calc_lambda_full(tau * lratio, mu, wmu, profile, wx)
    print(np.shape(LL))
    #plt.tight_layout()
    res = []
    res.append(fin)
    res.append(I_out)
    res.append(LL)
    res.append(rel_err)
    res.append(true_err)
    return res



# A basic example for testing purposes
S_g = solve_2level_nlte(91, 3, 21, 1.0, 1, lratio = 1E3, slab = True) # this is the solution which contains
S = S_g[0] # the source function
I = S_g[1] # the outgoing intensity
L = S_g[2] # the Lambda operator
rel_err = S_g[3] # relative error per iteration
true_err = S_g[4] # true error per iteration
# Plotting the Lambda operator 
plt.figure(constrained_layout = True, figsize = (8, 6))
plt.imshow(np.log10(L), origin = "lower", cmap = "gray")
plt.show()


# Plotting relative and true error (testing the convergence esentially)
plt.figure(constrained_layout = True, figsize = (8, 6))
plt.semilogy(rel_err, '--k', linewidth = 2, label = "Relative error")
plt.semilogy(true_err, color = "orange", linewidth = 2, label = "True error")
plt.xlabel('Iteration number', fontsize = 18)
plt.ylabel('Max. relative correction and true error', fontsize = 18)
plt.legend()
plt.show()

