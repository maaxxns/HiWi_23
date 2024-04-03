import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d

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

freq_ref = np.linspace(5*10**11, 3*10**12, 100) #test freq from 500 Ghz to 3 THz
n_test = np.linspace(4.2,4.4,100)
k_test = 1.1 * np.linspace(1,2,100)

T = Transfer_function_three_slabs(freq_ref, 1 , n_test, 1, 1, k_test, 1, d, True)

r_p = np.array([n_test[-1],k_test[-1]])

steps = np.linspace(1, 12000, 12000, dtype=int)

r_per_freq = [None]*(len(freq_ref))
r_per_step = [None]*(len(steps) + 1)
delta_per_freq = [None]*(len(freq_ref))
delta_per_step = [None]*(len(steps))
threshold_n = 0.1
threshold_k = 0.1
h = 0.06

epsilon = 10**-3
i = 0
H_0_value_reversed = reverse_array(T)
phase_rev = reverse_array(np.unwrap(np.angle(T)))
r_per_step[0] = r_p
kicker_n, kicker_k = 0.5, 0.5

for freq in tqdm(reverse_array(freq_ref[1:-1])): #walk through frequency range from upper to lower limit
    index = np.argwhere(freq_ref==freq)[0][0]
    params_delta_function = [H_0_value_reversed[index], phase_rev[index], np.array([freq_ref[index- 1], freq_ref[index], freq_ref[index + 1]]), Material]
    for step in steps:
        r_per_step[step] = newton_minimizer(delta_of_r, r_per_step[step - 1], params=params_delta_function, h = h)
        r_0 = r_per_step[step]
        delta_per_step[step - 1] = delta_of_r(r_per_step[step - 1], params_delta_function)
        if(np.abs(r_per_step[step][0] - r_per_step[step-1][0]) < epsilon and np.abs(r_per_step[step][1] - r_per_step[step-1][1]) < epsilon): #break condition for when the values seems to be fine
            break
        if(r_per_step[step][0] < threshold_n): # This is just a savety measure if the initial guess is not good enough
            r_per_step[step][0] = r_0[0] + kicker_n
            kicker_n = kicker_n + 0.5
            if(step > 100):
                step = step - 100 # every time we engage the kicker we give the algo a bit time
        if(r_per_step[step][1] < threshold_k):
            r_per_step[step][1] = r_0[1] + kicker_k
            kicker_k = kicker_k + 0.5
            if(step > 100):
                step = step - 100 # every time we engage the kicker we give the algo a bit time
    kicker_n, kicker_k = 0.5, 0.5 # reset kickers
    mask_delta_per_step = np.isfinite(np.array(delta_per_step).astype(np.double))
    delta_per_freq[index] = np.array(delta_per_step)[mask_delta_per_step]
    r_per_freq[index] = [r_0[0], r_0[1]] # save the final result of the Newton method for the frequency freq
    r_per_step[0] = r_0 # use the n and k value from the last frequency step as guess for the next frequency

#r_per_freq = reverse_array(r_per_freq) # we need to turn the array back around
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
#plt.title('parameter: epsilon ' + str(epsilon) + ', h ' + str(h) + ', kicker n ' + str(kicker_n) + ', kicker k' + str(kicker_k) + ', start r ' + str(r_p))
plt.legend()
plt.savefig('build/testing/test.pdf')

plt.close()
plt.figure()
plt.plot(range(len(delta_per_freq[1])), (delta_per_freq[1]), label='delta 1')
plt.plot(range(len(delta_per_freq[-2])), (delta_per_freq[-2]), label='delta - 1')
plt.plot(range(len(delta_per_freq[50])), (delta_per_freq[50]), label='delta 50')
plt.plot(range(len(delta_per_freq[80])), (delta_per_freq[80]), label='delta 80')

plt.xlabel(r'steps')
plt.ylabel('delta')
#plt.title('parameter: epsilon ' + str(epsilon) + ', h ' + str(h) + ', kicker n ' + str(kicker_n) + ', kicker k' + str(kicker_k) + ', start r ' + str(r_p))
plt.legend()
plt.savefig('build/testing/delta.pdf')
plt.close()

#########################################################################################################################################################
#       3 d wireframe plot of dela function
#########################################################################################################################################################


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#index = 80
#params_delta_function = [H_0_value_reversed[index], phase_rev[index], np.array([freq_ref[index- 1], freq_ref[index], freq_ref[index + 1]]), Material]
#
#
## Grab some test data.
#Z = np.empty(shape=(len(n_test),len(k_test)))
#for q in tqdm(range(len(n_test))):
#    for a in range(len(k_test)):
#        Z[q,a] = delta_of_r([n_test[q], k_test[a]], params_delta_function)
#
#X, Y= n_test, k_test
## Plot a basic wireframe.
#ax.plot_wireframe(X[0::5], Y[0::5], Z[0::5,0::5], rstride=10, cstride=10)
#
#plt.savefig('build/testing/delta3d.pdf')
#
#plt.close()

#########################################################################################################################################################
#########################################################################################################################################################


# Lets hava a look at the delta function
delta_test = np.empty(shape=len(n_test[1:-1]))
i = 0
for ns in (n_test[1:-1]):
    
    index = 80
    params_delta_function = [H_0_value_reversed[index], phase_rev[index], np.array([freq_ref[index- 1], freq_ref[index], freq_ref[index + 1]]), Material]
    delta_test[i] = delta_of_r([ns, 1.2], params_delta_function)
    i = i + 1
plt.figure()
plt.plot(n_test[1:-1], delta_test, label='delta')
plt.xlabel('n')
plt.ylabel('delta')
plt.savefig('build/testing/delta_func_for_n.pdf')
#########################################################################################################################################################

# for testing out the minimizer for different function
"""
def test_function(r, params):
    A = params[0]
    b = params[1] 
    c_ = params[2]
    return 1/2 * np.array(r).dot(A).dot(np.array(r)) - b.dot(np.array(r)) + c_ 

params = [np.array([[2,-2], [1,2]]), np.array([1,2]), 0]

for step in steps:
    r_per_step[step] = newton_minimizer(test_function, r_per_step[step - 1], params, h=0.5)
    i = i + 1
    if(np.abs(r_per_step[step][0] - r_per_step[step-1][0]) < epsilon and np.abs(r_per_step[step][1] - r_per_step[step-1][1]) < epsilon): #break condition for when the values seems to be fine
        break

plt.figure()
plt.plot(steps[1:i], flatten(r_per_step[1:i])[0::2], label='esti n') # we have to flatten the array before it plots 
plt.plot(steps[1:i], flatten(r_per_step[1:i])[1::2], label='esti k')
#plt.plot(freq_ref[minlimit:maxlimit]/1e12, alpha, label=r'$\alpha$')

plt.xlabel(r'$\omega / THz$')
plt.ylabel('value')
#plt.title('parameter: epsilon ' + str(epsilon) + ', h ' + str(h) + ', kicker n ' + str(kicker_n) + ', kicker k' + str(kicker_k) + ', start r ' + str(r_p))
plt.legend()
plt.savefig('build/testing/test.pdf')
plt.close()
B = [None] * i
for q in range(i):
    B[q] = test_function(r_per_step[q], params)
plt.plot(flatten(r_per_step[:i])[0::2], B, label="test" )
#plt.plot(freq_ref[minlimit:maxlimit]/1e12, alpha, label=r'$\alpha$')

plt.xlabel('n')
plt.ylabel('value')
#plt.title('parameter: epsilon ' + str(epsilon) + ', h ' + str(h) + ', kicker n ' + str(kicker_n) + ', kicker k' + str(kicker_k) + ', start r ' + str(r_p))
plt.legend()
plt.savefig('build/testing/test_func.pdf')
"""