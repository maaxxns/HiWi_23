import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *
from tqdm import tqdm

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

freq_ref = np.linspace(5*10**11, 3*10**12, 1000) #test freq from 500 Ghz to 3 THz
n_test = 1.2 * np.linspace(1,2,1000)
k_test = 1.1 * np.linspace(1,2,1000)

T = Transfer_function_three_slabs(freq_ref, 1 , n_test, 1, 1, k_test, 1, d, True)

r_p = np.array([n_test[0],k_test[0]])

steps = np.linspace(1, 12000, 12000, dtype=int)

r_per_freq = [None]*(len(freq_ref))
r_per_step = [None]*len(steps)
threshold_n = 0.1
threshold_k = 0.1
h = 0.065

epsilon = 10**-3
i = 0
H_0_value_reversed = reverse_array(T)
phase_rev = reverse_array(np.unwrap(np.angle(T)))
r_per_step[0] = r_p

for freq in tqdm(reverse_array(freq_ref[1:-1])): #walk through frequency range from upper to lower limit
    index = np.argwhere(freq_ref==freq)[0][0]
    params_delta_function = [H_0_value_reversed[index], phase_rev[index], np.array([freq_ref[index- 1], freq_ref[index], freq_ref[index + 1]]), Material]
                                        ##### not sure if this works
    for step in steps:
        r_per_step[step] = newton_minimizer(delta_of_r, r_per_step[step - 1], params=params_delta_function, h = h) # why is the convergence of my newton linear?
        r_0 = r_per_step[step]
        if(np.abs(r_per_step[step][0] - r_per_step[step-1][0]) < epsilon and np.abs(r_per_step[step][1] - r_per_step[step-1][1]) < epsilon): #break condition for when the values seems to be fine
            break
        if(r_per_step[step][0] < threshold_n): # This is just a savety measure if the initial guess is not good enough
            r_per_step[step][0] = r_0[0] + kicker_n
            kicker_n = kicker_n + 0.5
            #print("kicker used for n, kicker at: ", kicker_n)
            if(step > 100):
                step = step - 100 # every time we engage the kicker we give the algo a bit time
        if(r_per_step[step][1] < threshold_k):
            r_per_step[step][1] = r_0[1] + kicker_k
            kicker_k = kicker_k + 0.5
            #print("kicker used for k, kicker at: ", kicker_k)
            if(step > 100):
                step = step - 100 # every time we engage the kicker we give the algo a bit time
    kicker_n, kicker_k = 0.5, 0.5 # reset kickers
    r_per_freq[index] = [r_0[0], r_0[1]] # save the final result of the Newton method for the frequency freq
    r_per_step[0] = r_0 # use the n and k value from the last frequency step as guess for the next frequency

r_per_freq = reverse_array(r_per_freq) # we need to turn the array back around

print("Done")
print("Plotting...")
plt.figure()
plt.plot(freq_ref[1:-2]/1e12, flatten(r_per_freq[1:-2])[0::2], label='esti n') # we have to flatten the array before it plots 
plt.plot(freq_ref[1:-2]/1e12, flatten(r_per_freq[1:-2])[1::2], label='esti k')
plt.plot(freq_ref[1:-2]/1e12, n_test[1:-2], label='actual n')
plt.plot(freq_ref[1:-2]/1e12, k_test[1:-2], label='actual k')

#plt.plot(freq_ref[minlimit:maxlimit]/1e12, alpha, label=r'$\alpha$')

plt.xlabel(r'$\omega / THz$')
plt.ylabel('value')
plt.title('parameter: epsilon ' + str(epsilon) + ', h ' + str(h) + ', kicker n ' + str(kicker_n) + ', kicker k' + str(kicker_k) + ', start r ' + str(r_p))
plt.legend()
plt.savefig('build/testing/test.pdf')
