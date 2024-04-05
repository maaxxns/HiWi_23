import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *
from tqdm import tqdm
from mpl_toolkits.mplot3d import axes3d

"""
Problems to be solved:
    - Unwrapping with just three complex angles doesnt work because if the angles are pos-neg-neg or neg-pos-pos the unwrapping 
      cant see the bigger picture for example in the actual array the angles might look like pos-pos-pos-pos-neg-neg so the double neg should be
      flipped up instead of the one pos before them being flipped down. This results in a "jump" that shouldnt happen.
      Meaning I have to find a better way for the unwrapping.

    Idea to solve it:
    1. -Runtime effiecient- maybe I can take an array that is a bit bigger but not the whole frequency range like 10-50 array entrances for unwrapping.
    This would already increase computation time by a factor of atleast 5-25.
    2. -brute force method- Take the whole frequency range for unwrapping. This would take insanely long for one computation step.
    3. find a better unwrapping 

    for now I use solution 2

    Problem 2
    - Noise seems to be a problem if its introcude into the Transmittion function somehow?
        I still get good results when I just introduce noise into the n and k value


"""

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

freq_ref = np.linspace(5*10**11, 3*10**12, 300) #test freq from 500 Ghz to 3 THz
noise = np.random.normal(0,0.00001,len(freq_ref)) + 1j*np.random.normal(0,0.00001,len(freq_ref))
n_test = 0.5*np.exp(0.2*np.linspace(1,10,300)) + 3
k_test = 1.1 * np.linspace(1,5,300)

T = Transfer_function_three_slabs(freq_ref, 1 , n_test, 1, 1, k_test, 1, d, True) + noise


T = T

r_p = np.array([estimater_n(np.unwrap(np.angle(T)), freq_ref, Material)[-1],10])

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
H_0_value = T
phase = np.unwrap(np.angle(T))

plt.figure()
plt.plot(freq_ref, np.unwrap(np.angle(T)), label='unwrapped')
plt.plot(freq_ref, np.angle(T), label='unwrapped')
plt.legend()
plt.xlabel("freq")
plt.ylabel('phase')
plt.savefig('build/testing/phase.pdf')
plt.close()


r_per_step[0] = r_p
kicker_n, kicker_k = 0.5, 0.5

for freq in tqdm(reverse_array(freq_ref[1:-1])): #walk through frequency range from upper to lower limit
    index = np.argwhere(freq_ref==freq)[0][0]
    params_delta_function = [H_0_value[index], phase[index], freq_ref, index,  Material]
    for step in steps:
        r_per_step[step] = newton_minimizer(delta_of_r_whole_frequency_range, r_per_step[step - 1], params=params_delta_function, h = h)
        r_0 = r_per_step[step]
        delta_per_step[step - 1] = delta_of_r_whole_frequency_range(r_per_step[step - 1], params_delta_function)
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
plt.title('noise std' + str(0.00001) +'parameter: epsilon ' + str(epsilon) + ', h \n' + str(h) + ', kicker n ' + str(kicker_n) + ', kicker k' + str(kicker_k) + ', start r ' + str(r_p))
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
#params_delta_function = [H_0_value[index], phase[index], freq_ref, index, Material]
#
#
## Grab some test data.
#Z = np.empty(shape=(len(n_test),len(k_test)))
#for q in tqdm(range(len(n_test))):
#    for a in range(len(k_test)):
#        Z[q,a] = delta_of_r_whole_frequency_range([n_test[q], k_test[a]], params_delta_function)
#
#X, Y= n_test, k_test
## Plot a basic wireframe.
#ax.plot_wireframe(X[0::5], Y[0::5], Z[0::5,0::5], rstride=10, cstride=10)
#
#plt.savefig('build/testing/delta3d.pdf')

plt.close()

#########################################################################################################################################################
#########################################################################################################################################################

# Plots the delta function for one freq value and one k value but differet n values
# Lets hava a look at the delta function
delta_test = np.empty(shape=len(n_test[1:-1]))
i = 0
for ns in (n_test[1:-1]):
    
    index = 80
    params_delta_function = [H_0_value[index], phase[index], freq_ref, index, Material]
    delta_test[i] = delta_of_r_whole_frequency_range([ns, 1.2], params_delta_function)
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