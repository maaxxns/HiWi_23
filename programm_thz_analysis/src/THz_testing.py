import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *

@dataclass
class Material_parameters:
    d: float
    n_1: float
    k_1: float 
    n_3: float 
    k_3: float

# The thickness of the probe

d = 0.26*10**(-3) # thickness of the probe in SI
n_air = 1
n_slab = 1
k_slab = 1




Material = Material_parameters(d = d, n_1=n_air, k_1=n_air, n_3=n_slab, k_3=k_slab)

freq = np.linspace(1*10**10, 3*10**12, 1000) #test freq from 10 Ghz to 3 THz


T = Transfer_function_three_slabs(freq, 1 , 2.5, 1, 1, 2.5, 1, d, True)

r_p = np.array([1,1])

steps = np.linspace(1, 12000, 12000, dtype=int)
params = [T[100], freq[100], Material]

r = [None]*(len(steps) + 1)
ns = [None]*len(r)
ks = [None]*len(r)
ns[0] = r_p[0]
ks[0] = r_p[1]

r[0] = r_p
epsilon = 10**-6
i = 0
for step in steps:
    r[step] = newton_minimizer(delta_of_r, r[step - 1], params=params, h = 0.065) # why is the convergence of my newton linear?
    #print(r[step])
    ns[step] = r[step][0]
    ks[step] = r[step][1]
    i = i + 1
    if(np.abs(r[step][0] - r[step-1][0]) < epsilon and np.abs(r[step][1] - r[step-1][1]) < epsilon):
        break

x = np.linspace(0, i, i)
plt.figure()
plt.plot(x, ns[:i], label='n')
plt.plot(x, ks[:i], label='k')
plt.legend()
plt.xlabel('steps')
plt.ylabel('value')
plt.savefig('build/testing/convergence.pdf')


nns = np.linspace(0.5, 4, 1000)
delta = [None] * len(nns)
i = 0
for nn in nns:
    delta[i] = delta_of_r([nn, 2], params)
    i = i + 1

plt.figure()
plt.plot(nns, delta, label='delta')
plt.legend()
plt.xlabel('n')
plt.ylabel('delta')
plt.savefig('build/testing/delta.pdf')

