import matplotlib.pyplot as plt 
import numpy as np 
import astropy.io.fits as fits
import sys
from __future__ import print_function

#-------------------------------------------
# Short characteristics formal solver:

def sc_formal_solver(I_upwind,delta,S_upwind,S_local):

    expd = np.exp(-delta)
    w_local = 1.0 - 1.0/delta * (1.0 - expd)
    w_upwind = -expd + 1.0/delta * (1.0 - expd)

    I_local = I_upwind * expd + \
    w_local * S_local + w_upwind * S_upwind

    return I_local

def sc_formal_solver_alo(I_upwind,delta,S_upwind,S_local):

    expd = np.exp(-delta)
    w_local = 1.0 - 1.0/delta * (1.0 - expd)
    w_upwind = -expd + 1.0/delta * (1.0 - expd)

    I_local = I_upwind * expd + \
    w_local * S_local + w_upwind * S_upwind

    return I_local, w_local




#-------------------------------------------

# We will define a discrete grid for our spacial coordinate, that is logtau 

ND = 81
logtau = np.linspace(-5,3,ND)
tau = 10.0**logtau

B = np.zeros(ND)
B[:] = 1.0

eps = np.zeros(ND)
eps[:] = 1E-2

S = np.copy(B)
local_lambda_operator = np.zeros(ND)

I_plus = np.zeros(ND)
I_minus = np.zeros(ND)

#Do one formal solution upward:
I_plus[ND-1] = S[ND-1]
local_lambda_operator[ND-1] = 1.0
for d in range (ND-2,-1,-1):
    I_plus[d], local_alo  = sc_formal_solver_alo(I_plus[d+1],tau[d+1]-tau[d],S[d+1],S[d])
    local_lambda_operator[d] = local_alo


#Do one formal solution downward:
I_minus[0] = 0.0
for d in range (1,ND):
    I_minus[d], local_alo = sc_formal_solver_alo(I_minus[d-1],tau[d]-tau[d-1],S[d-1],S[d])
    local_lambda_operator[d] += local_alo

local_lambda_operator[:] *= 0.5

#plot them both to see how they look like:

plt.figure(figsize=[4,3])
plt.plot(logtau,I_plus,label='outgoing')
plt.plot(logtau,I_minus,label='incoming')
plt.xlabel('$\log\\tau$')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.savefig('iteration_one.png',bbox_inches='tight')

# then update the source function

J = (I_minus + I_plus)*0.5

S = eps * B + (1.0-eps)*(I_minus + I_plus) * 0.5
#S = (eps * B + (1.0-eps) * (J- local_lambda_operator * S)) / (1.0 - (1.0-eps)*local_lambda_operator)

# plot that too:
plt.clf()
plt.cla()
plt.figure(figsize=[4,3])
plt.plot(logtau,S)
plt.xlabel('$\log\\tau$')
plt.ylabel('Source function')
plt.legend()
plt.tight_layout()
plt.savefig('S_iteration_one.png',bbox_inches='tight')


#Now we will iteratively correct:
N_iter = 1000

# Here we track relative difference
rel_diff = np.zeros(N_iter)

for i in range(0,N_iter):
    I_plus[ND-1] = S[ND-1]
    local_lambda_operator[ND-1] = 1.0
    for d in range (ND-2,-1,-1):
        I_plus[d], local_alo  = sc_formal_solver_alo(I_plus[d+1],tau[d+1]-tau[d],S[d+1],S[d])
        local_lambda_operator[d] = local_alo

    I_minus[0] = 0.0
    for d in range (1,ND):
        I_minus[d], local_alo = sc_formal_solver_alo(I_minus[d-1],tau[d]-tau[d-1],S[d-1],S[d])
        local_lambda_operator[d] += local_alo

    local_lambda_operator[:] *= 0.5

    S_old = S #to evaluate change
    J = (I_minus + I_plus)*0.5
    S = eps * B + (1.0-eps)*(I_minus + I_plus) * 0.5
    #S = (eps * B + (1.0-eps) * (J- local_lambda_operator * S)) / (1.0 - (1.0-eps)*local_lambda_operator)
    
    plt.plot(logtau,S)
    rel_diff[i] = np.abs((S[0]-S_old[0]) / S_old[0])
    print('Iteration #', i)
    if (rel_diff[i] < 1E-5):
        break

print(I_plus[0])

plt.savefig('S_change.png')

# Plot the final source function in log scale
plt.cla()
plt.figure(figsize=[4,3])
plt.plot(logtau,np.log10(S))
plt.xlabel('$\log \\tau$')
plt.ylabel('$\log S$')
plt.tight_layout()
plt.savefig('final_S.png',bbox_inches='tight')

# Evolution of the relative change is much better seen in log scale:

# Plot the evolution of the relative change
plt.clf()
plt.cla()
plt.figure(figsize=[4,3])
plt.plot(rel_diff)
plt.xlabel('Iteration #')
plt.ylabel('Relative difference')
plt.tight_layout()
plt.savefig('convergence.png',bbox_inches='tight')

# Evolution of the relative change is much better seen in log scale:
plt.clf()
plt.cla()
plt.figure(figsize=[4,3])
plt.plot(np.log10(rel_diff))
plt.xlabel('Iteration #')
plt.ylabel('Log Relative difference')
plt.tight_layout()
plt.savefig('log_convergence.png',bbox_inches='tight')