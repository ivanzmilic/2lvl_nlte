import matplotlib.pyplot as plt 
import numpy as np 
import astropy.io.fits as fits
from numba import jit
import sys

font = {'family' : 'normal',                                   
        'weight' : 'normal',             
        'size'   : 18} 

import matplotlib
matplotlib.rc('font', **font)  

# -----------------------------------------------------------------------------------
# A simple demo for 2-level atom NLTE spectral line formation
# Written by Ivan Milic (CU/LASP/NSO) with help and contribution from C. Osborne (U Glasgow)
# The code starts around line ~ 150

#-------------------------------------------
# Short characteristics formal solver:
@jit(nopython=True)
def sc_2nd_order(tau, S, mu, I_boundary):

    #first we determine direction:
    ND = S.shape[0]
    begin = ND-1
    end = -1
    step = -1
    if (mu<0):
        begin = 0
        end = S.shape[0]
        step = 1

    I = np.zeros(ND)
    L = np.zeros(ND)
    I[begin] = I_boundary
    #L[begin] = I[begin] / S[begin]
    
    for d in range(begin+step,end-step,step):
        
        delta_u = (tau[d-step] - tau[d])/mu
        delta_d = (tau[d] - tau[d+step])/mu

        expd = np.exp(-delta_u)
        if delta_u <= 0.01:
            w0=delta_u*(1.-delta_u/2.+delta_u**2/6.-delta_u**3/24.+delta_u**4/120.-delta_u**5/720.+delta_u**6/5040.-delta_u**7/40320.+delta_u**8/362880.)
            w1=delta_u**2*(0.5-delta_u/3.+delta_u**2/8.-delta_u**3/30.+delta_u**4/144.-delta_u**5/840.+delta_u**6/5760.-delta_u**7/45360.+delta_u**8/403200.)
            w2=delta_u**3*(1./3.-delta_u/4.+delta_u**2/10.-delta_u**3/36.+delta_u**4/168.-delta_u**5/960.+delta_u**6/6480.-delta_u**7/50400.+delta_u**8/443520.)
        else:
            w0 = 1.0 - expd
            w1 = w0 - delta_u * expd
            w2 = 2.0 * w1 - delta_u * delta_u * expd

        psi0 = w0 + (w1 * (delta_u/delta_d - delta_d / delta_u) - w2 * (1.0 / delta_d + 1.0 / delta_u)) / (delta_u + delta_d)
        psiu = (w2 / delta_u + w1*delta_d/delta_u)/(delta_u+delta_d)
        psid = (w2 / delta_d - w1 * delta_u/delta_d)/(delta_u+delta_d)

        I[d] = I[d-step]*expd + psiu*S[d-step] + psi0*S[d] + psid*S[d+step]
        L[d] = psi0
    #last point is linear:

    d = end-step
    delta_u = (tau[d-step] - tau[d])/mu
    expd = np.exp(-delta_u)
    
    psi0 = 1.0 - 1.0/delta_u * (1.0 - expd)
    psiu = -expd + 1.0/delta_u * (1.0 - expd)

    if (delta_u < 0.01):
        expd = 1.0 - delta_u + delta_u**2.0 / 2.0 - delta_u**3.0/6.0
        psi0 = delta_u/2. - delta_u*delta_u/6. + delta_u**3.0 / 24.
        psiu = delta_u/2. - delta_u**2.0/3. + delta_u**3.0 / 8.

    I[d] = I[d-step]*expd + psiu*S[d-step] + psi0*S[d]
    L[d] = psi0

    return np.stack((I,L))

@jit(nopython=True)
def calc_lambda_full(tau, mu, wmu, profile, wx):

    ND = tau.shape[0]
    Lambda_full = np.zeros((ND,ND))

    for d in range(0,ND):
        S_mock = np.zeros(ND)
        S_mock[d] = 1.0
        for m in range(0,mu.shape[0]):
            for l in range(0,profile.shape[0]):

                #outward:
                Lambda_monoc = sc_2nd_order(tau*profile[l],S_mock,mu[m],0.)

                Lambda_full[:,d]+=Lambda_monoc[0]*profile[l]*wx[l]*wmu[m]*0.5
                
                #inward
                Lambda_monoc = sc_2nd_order(tau*profile[l],S_mock,-mu[m],0.)

                Lambda_full[:,d]+=Lambda_monoc[0]*profile[l]*wx[l]*wmu[m]*0.5

    return Lambda_full

@jit(nopython=True)
def calc_lambda_monoc(tau, mu):
    ND = tau.shape[0]
    lambda_monoc = np.zeros((ND,ND))

    for d in range(0,ND):
        S_mock = np.zeros(ND)
        S_mock[d] = 1.0
            
        lambda_monoc[:,d] = sc_2nd_order(tau,S_mock,mu,0.)[0]

    return lambda_monoc

@jit(nopython=True)
def calc_anisotropy(tau,mu,wmu,profile,wx,S):

    ND = tau.shape[0]
    J_02 = np.zeros(ND)

    for m in range(0,mu.shape[0]):
        for l in range(0,profile.shape[0]):

            #outward:
            I = sc_2nd_order(tau*profile[l],S,mu[m],S[-1])
            J_02+=I[0]*profile[l]*wx[l]*wmu[m]*0.5*(1.0-3.*mu[m]*mu[m]) /2.4248
            
            #inward
            I = sc_2nd_order(tau*profile[l],S,-mu[m],0.)
            J_02+=I[0]*profile[l]*wx[l]*wmu[m]*0.5*(1.0-3.*mu[m]*mu[m]) /2.4248
    
    return J_02



#@jit('float64[:,:](float64[:],float64[:],float64[:],float64[:],float64[:],float64[:,:])')
#def calc_anisotropy_response(tau,mu,wmu,profile,wx,dS_dB):


@jit(nopython=True)
def one_full_fs(tau,S,mu,profile,boundary):

    NL = profile.shape[0]
    I = np.zeros(NL)

    for l in range(0,NL):
        I[l] = sc_2nd_order(tau*profile[l],S,mu,boundary)[0,0]

    return I