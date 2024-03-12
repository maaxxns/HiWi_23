import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *
###################################################################################################################################
# Datatypes

@dataclass
class Material_parameters:
    d: float
    n_1: float
    k_1: float 
    n_3: float 
    k_3: float

###################################################################################################################################
###################################################################################################################################
#       All data is read in, in this block.
#       Necessary data are
#       - the time resolved THz measurment
#       - the thickness of the probe
###################################################################################################################################


# The thickness of the probe

d = 0.26*10**(-3) # thickness of the probe in SI
n_air = 1
n_slab = 1
k_slab = 1

Material = Material_parameters(d = d, n_1=n_air, k_1=n_air, n_3=n_slab, k_3=k_slab)

#Read the excel file
material_properties_ref = np.genfromtxt('../data/teflon_1_material_properties.txt')
data_sam = np.genfromtxt('../data/without_cryo_with_purge_teflon.txt', delimiter="	", comments="#") # The time resolved dataset of the probe measurment
data_ref = np.genfromtxt('../data/without_cryo_with_purge.txt',  delimiter="	", comments="#") # the time resolved dataset of the reference measurment


###################################################################################################################################
# This block plots the raw dataset
###################################################################################################################################
ones = np.ones(10000)

data_ref[:,0] = data_ref[:,0] * 10**(-12) # ps in seconds
data_sam[:,0] = data_sam[:,0] * 10**(-12)

data_ref[:,0] = data_ref[:,0] + np.abs(np.min(data_ref[:,0])) # move everything to positiv times
data_sam[:,0] = data_sam[:,0] + np.abs(np.min(data_sam[:,0]))

#plot the intensity against time delay
plt.figure()
plt.plot(data_ref[:,0]*10**(12), data_ref[:,1], label='Reference')
#plt.vlines(data_ref[peaks_inref,0]*10**12, ymin=0, ymax=-10)
plt.plot(data_sam[:,0]*10**(12) + 100, data_sam[:,1], label='Sample moved by +100ps')
plt.xlabel(r'$ t/ps $')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.title('The reference and sample data set of silicon')
plt.savefig('build/THz_timedomain.pdf')
plt.close()


##################################################################################################################################
# Zero_padding
##################################################################################################################################
timestep = np.abs(data_ref[:,0][2]-data_ref[:,0][3]) # minimum time resolution
N = len(data_ref[:,0]) #number of total data points

num_zeros = 1500

peak_ref,prop = find_peaks(data_ref[:,1], prominence=0.3) # finds the highest peak in the dataset and returns its index
peak_ref = peak_ref[0:2]
peak_sam,prop = find_peaks(data_sam[:,1], prominence=0.5)
peak_sam = peak_sam[0:2]
peak_sam[1] = peak_sam[1] - 100 # we assume that we cut off the array 50 steps before we hit the post pulse

data_ref_zero = [np.append(data_ref[:peak_sam[1], 0], np.linspace(data_ref[peak_sam[1], 0], data_ref[peak_sam[1], 0]+num_zeros*timestep, num_zeros)),
                 np.append(data_ref[:peak_sam[1], 1], (np.zeros(num_zeros)))]
data_sam_zero = [np.append(data_sam[:peak_sam[1], 0], np.linspace(data_sam[peak_sam[1], 0], data_sam[peak_sam[1], 0]+num_zeros*timestep, num_zeros)),
                 np.append(data_sam[:peak_sam[1], 1], (np.zeros(num_zeros)))]

print(len(data_ref_zero[0]))

##################################################################################################################################
# Plot zero padding in time domain
##################################################################################################################################

plt.figure()
plt.plot(data_ref_zero[0]*10**(12), data_ref_zero[1], label='Reference')
#plt.vlines(data_ref[peaks_inref,0]*10**12, ymin=0, ymax=-10)
plt.plot(data_sam_zero[0]*10**(12) + 100, data_sam_zero[1], label='Sample moved by +100ps')
plt.xlabel(r'$ t/ps $')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.title('The reference and sample data set of silicon with zero padding')
plt.savefig('build/THz_timedomain_zero.pdf')
plt.close()


##################################################################################################################################
# Some necessary calculations on frequency and time resolution
##################################################################################################################################

Delta_f = 1/(N*timestep) #frequency resolution
print('Delta t = ', " ",timestep/10**(-12), "ps")
print("T ", N*timestep/10**(-12), "ps")
print("Delta f = ", " ", Delta_f*10**(-12), "THz")

peak_ref,prop = find_peaks(data_ref[:,1], prominence=1) # finds the highest peak in the dataset and returns its index
peak_ref = peak_ref[0]
peak_sam,prop = find_peaks(data_sam[:,1], prominence=1)
peak_sam = peak_sam[0]

t_peak_ref = data_ref[peak_ref,0] # the time value of the highest peak in the dataset
t_peak_sam = data_sam[peak_sam,0]
###################################################################################################################################
# This block applies the FFT to the data, aswell as masking frequencies that we dont need for the analization
###################################################################################################################################

freq_ref, amp_ref = FFT_func(data_ref[:,1], data_ref[:,0])  #in Hz
freq_sam, amp_sam = FFT_func(data_sam[:,1], data_sam[:,0])

mask1 = freq_ref < 4.5*10**12 # mask1ed for THz frequency below 4.5 THz
amp_ref = amp_ref[mask1]
amp_sam = amp_sam[mask1]
freq_ref = freq_ref[mask1]
freq_sam = freq_sam[mask1]

###################################################################################################################################
# This block applies the FFT to the zero padded data, aswell as masking frequencies that we dont need for the analization
###################################################################################################################################
freq_ref_zero, amp_ref_zero = FFT_func(data_ref_zero[1], data_ref_zero[0])  #in Hz
freq_sam_zero, amp_sam_zero = FFT_func(data_sam_zero[1], data_sam_zero[0])

mask1_zero = freq_ref_zero < 4.5*10**12 # mask1_zero masks for THz frequency below 4.5 THz
amp_ref_zero = amp_ref_zero[mask1_zero]
amp_sam_zero = amp_sam_zero[mask1_zero]
freq_ref_zero = freq_ref_zero[mask1_zero]
freq_sam_zero = freq_sam_zero[mask1_zero]

##########################################################################################
# This block plots the FFT
##########################################################################################

plt.figure()
plt.plot(freq_ref* 10**(-12), np.abs(amp_ref), label='Reference FFT') # plot in Thz
plt.plot(freq_sam* 10**(-12), np.abs(amp_sam), label='Sample FFT')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
plt.title('The FFT of the data sets of silicon')
plt.savefig('build/THz_FFT.pdf')
plt.close()

##########################################################################################
# This block plots the FFT of the zerp padded data
##########################################################################################

plt.figure()
plt.plot(freq_ref_zero* 10**(-12), np.abs(amp_ref_zero), label='Reference FFT') # plot in Thz
plt.plot(freq_sam_zero* 10**(-12), np.abs(amp_sam_zero), label='Sample FFT')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
plt.title('The FFT of the zero padded data sets of silicon')
plt.savefig('build/THz_FFT_zero.pdf')
plt.close()


###################################################################################################################################
# This block calculates the complex transfer function and does the unwrapping porcess
# It also plots the angle and phase of the transferfunction
###################################################################################################################################

H_0_value = H_0(amp_ref, amp_sam) # complex transfer function

angle = np.angle(H_0_value) #angle between complex numbers
phase = np.unwrap(angle)  #phase 

#phase_dif = unwrapping_alt(amp_ref, amp_sam, freq_ref)

# Note that phase and phase_dif should be equal. However, there are not. So either the paper is wrong. Or I am doing something wrong

plt.figure()
#plt.plot(freq_ref*10**(-12),phase_dif, label='phase differenz')
plt.plot(freq_ref*10**(-12),angle, label='angle directly from H_0')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\Phi$')
plt.legend()
plt.grid()
plt.title('The phase unwarpped')
plt.savefig('build/THzPhase.pdf')
plt.close()

###################################################################################################################################
# This block calculates the complex transfer function and does the unwrapping porcess
# It also plots the angle and phase of the transferfunction
###################################################################################################################################

H_0_value_zero = H_0(amp_ref_zero, amp_sam_zero) # complex transfer function

angle_zero = np.angle(H_0_value_zero) #angle between complex numbers
phase_zero = np.unwrap(angle_zero)  #phase 

#phase_dif = unwrapping_alt(amp_ref, amp_sam, freq_ref)

# Note that phase and phase_dif should be equal. However, there are not. So either the paper is wrong. Or I am doing something wrong

plt.figure()
#plt.plot(freq_ref_zero*10**(-12),angle_zero, label='angle')
plt.plot(freq_ref_zero*10**(-12),phase_zero, label='phase directly from H_0')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\Phi$')
plt.legend()
plt.grid()
plt.title('The phase unwarpped of the zero padded data of silicon')
plt.savefig('build/THzPhase_zero.pdf')
plt.close()


###################################################################################################################################
#   This block calculates the real and complex part of the refractive index
###################################################################################################################################

n_real = n(freq_ref, d, phase)
#n_real_alt = n(freq_ref, d, phase_dif)
n_im = k(freq_ref, d, H_0_value, n_real)

###################################################################################################################################
#   This block calculates the real and complex part of the refractive index
###################################################################################################################################

n_real_zero = n(freq_ref_zero, d, phase_zero)
#n_real_alt = n(freq_ref, d, phase_dif)
n_im_zero = k(freq_ref_zero, d, H_0_value_zero, n_real_zero)


###################################################################################################################################
#   This block plots the real part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), n(freq_ref, d, phase_dif), label='real part of refractive index calculated by the alternative unwrap')
plt.plot(freq_ref*10**(-12), n_real, label='real part of refractive index')
plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,1], c='k', label='refference n from PCA program')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
plt.ylim(0,2)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index of silicon')
plt.savefig('build/THz_real_index.pdf')
plt.close()

###################################################################################################################################
#   This block plots the real part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), n(freq_ref, d, phase_dif), label='real part of refractive index calculated by the alternative unwrap')
plt.plot(freq_ref_zero*10**(-12), n_real_zero, label='real part of refractive index')
plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,1], c='k', label='refference n from PCA program')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,4)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index from zero padded data of silicon')
plt.savefig('build/THz_real_index_zero.pdf')
plt.close()


##################################################################################################
#   This block plots the complex part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), k(freq_ref, d, H_0_value, n_real_alt), label='complex part of refractive index by alt')
plt.plot(freq_ref*10**(-12), n_im, label='complex part of refractive index')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('k (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,2000)
plt.legend()
plt.grid()
plt.title('The complex part of the refractive index of silicon')
plt.savefig('build/THz_complex_index.pdf')
plt.close()

##################################################################################################
#   This block plots the complex part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), k(freq_ref, d, H_0_value, n_real_alt), label='complex part of refractive index by alt')
plt.plot(freq_ref_zero*10**(-12), n_im_zero, label='complex part of refractive index')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('k (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,2000)
plt.legend()
plt.grid()
plt.title('The complex part of the refractive index from zero padded data of silicon')
plt.savefig('build/THz_complex_index_zero.pdf')
plt.close()

###################################################################################################################################
# This block calculates the absorption coefficient and plots it
###################################################################################################################################

alpha = 2*freq_ref *n_im/c 

plt.figure()
plt.plot(freq_ref*10**(-12), alpha/100, label='Absorption coefficient')
plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], c='k', label='refference k from PCA program')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 4.5)
#plt.ylim(0, 500)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient of silicon')
plt.savefig('build/THz_absorption.pdf')
plt.close()

###################################################################################################################################
# This block calculates the absorption coefficient for zero padding and plots it
###################################################################################################################################

alpha_zero = 2*freq_ref_zero*n_im_zero/c

plt.figure()
plt.plot(freq_ref_zero*10**(-12), alpha_zero/100, label='Absorption coefficient')
plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], c="k",label='refference k from PCA program')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 4.5)
#plt.ylim(0, 500)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient of silicon for zero padded data')
plt.savefig('build/THz_absorption_zero.pdf')
plt.close()

###################################################################################################################################

###################################################################################################################################
# Here Starts the numerical process of finding the refractive index
###################################################################################################################################

###################################################################################################################################
""" For an actual calculation of n and k we would want to do it over the whole measured frequency range.
    However, this will used up way too much calculation power for my litle laptop so I start with just one frequency.
    If I get that stuff right I move on to the whole frequency range."""

print("-----------------------------------------------------------")

""""Okay ich glaub was ich eigentlich machen muss ist ein startwert zu wählen.
    Den nutze ich um eine Abweichung also delta zu berechnen.
    Mit dem Wert mache ich zudem dann den Newton schritt.
    Die Hesse Matrix berechne ich dabei an dem entsprechenden Punkt.
    Jetzt ist die Frage nur noch wie ich die Hesse Matrix berechne """


""" Die Transferfunktion gibt mir irgendwie keine sinnvollen werte zurück also plotte ich die erstmal für ein paar testwerte
    komischer weise scheint sie gegen hohe frequezen zu divergieren.
    
    Dieses Problem scheint zu verschwinden wenn ich die Fabry perot faktoren beachte.
    Allerdings stoße ich weiterhin auch divergenzen für bestimmte kombinationen aus n,k und omega."""

plt.figure()
plt.plot(freq_ref, Transfer_function_three_slabs(freq_ref, 1 , 2, 1, 1, 2, 1, d, True).real, label='Transferfunction real part')
plt.plot(freq_ref, Transfer_function_three_slabs(freq_ref, 1 , 2, 1, 1, 2, 1, d, True).imag, label='Transferfunction imag part')
plt.legend()
plt.savefig('build/testing/Transferfunction_n_2_k_2.pdf')
plt.close()

test_n = np.linspace(0,10, 1000)
test_k = np.linspace(0,10, 1000)

test_freq_index = 20

plt.figure()
plt.plot(test_n, Transfer_function_three_slabs(freq_ref[test_freq_index], 1 , test_n, 1, 1, 2, 1, d, True).real, label='Transferfunction n real part')
plt.plot(test_n, Transfer_function_three_slabs(freq_ref[test_freq_index], 1 , test_n, 1, 1, 2, 1, d, True).imag, label='Transferfunction n complex part')
plt.plot(test_n, Transfer_function_three_slabs(freq_ref[test_freq_index], 1 , 2, 1, 1, test_k, 1, d, True).real, label='Transferfunction k real part')
plt.plot(test_n, Transfer_function_three_slabs(freq_ref[test_freq_index], 1 , 2, 1, 1, test_k, 1, d, True).imag, label='Transferfunction k complex part')

plt.legend()
plt.title("Transferfunction at set frequency but different n and k")
plt.savefig('build/testing/Transferfunction_n_k.pdf')
plt.close()


###################################################################################################################################
# Set starting values for the algorythm
###################################################################################################################################
n_0 = 5
k_0 = 0.5


###################################################################################################################################
    # Finding the zero crossing of the f(r_p, omega) = T_calc - T_meas with newton
###################################################################################################################################

steps = 10
r_0 = np.array([n_0,k_0]) # r_p[0] = n, r_p[1] = k
r_p = r_0 # set the start value
r_p_1 = np.zeros((steps,2), dtype=complex)
r_p_1[0] = r_p
i = 1
h = 0.5
delta_values = np.zeros(steps)

n_1 = 1
n_3 = 1
k_1 = 1
k_3 = 1
fp = True
params_Transferfunction = [freq_ref[test_freq_index], n_1, n_3, k_1, k_3, d, fp]
params_delta_function = [H_0_value[test_freq_index], freq_ref[test_freq_index], Material]
""""
def f(r_p, params): # we try to find zero of this function
    print(r_p)
    print(Transfer_function_three_slabs(params[0], params[1] , r_p[0], params[2], params[3], r_p[1], params[4], params[5], params[6]) - H_0_value[test_freq_index])
    return Transfer_function_three_slabs(params[0], params[1] , r_p[0], params[2], params[3], r_p[1], params[4], params[5], params[6]) - H_0_value[test_freq_index]

for l in range(steps - 1):
    r_p_1[i] = newton_r_p_zero_finder(f, r_p, parameter = params_Transferfunction, h=h)
    r_p = r_p_1[i]
    delta_values[i] = delta_of_r(r_p, params_delta_function)
    print(f"{l/steps * 100:.2f} % {r_p}", end="\r")
    i = i + 1 
"""

"""
    - Why is it that whenever I try to calculate the Transferfunction it gives back a nan?
    - Why is it that the r_p becomes complex at some point and at that point the Transferfunction also becomes nan?
    - Another problem lies within the result that the Transferfunction is complex, but r_p should be complex
"""

###################################################################################################################################
    # Minimizing the delta function with newton 
###################################################################################################################################

steps = 20000

r_0 = np.array([n_0,k_0]) # r_p[0] = n, r_p[1] = k
r_p = r_0 # set the start value
r_p_1 = np.zeros((steps,2))
i = 1
h = 0.1
delta_values = np.zeros(steps)
for l in range(steps - 1):
    r_p_1[i] = newton_minimizer(delta_of_r, r_p, params_delta_function, h = h) # r_p[0] = n, r_p[1] = k
    delta_values[i] = delta_of_r(r_p, params_delta_function)
    if(r_p_1[i][0] == r_p_1[i - 1][0] or r_p_1[i][1] == r_p_1[i -1][1]): # if the algorythm brakes we go back two steps and try again with smaller h
        h = h/10
        i = i - 2
    r_p = r_p_1[i]
    print(f"{l/steps * 100:.2f} % {r_p}", end="\r")
    i = i + 1 

plt.figure
plt.plot(np.linspace(1,steps - 2, steps - 3), r_p_1[1:steps-2,0], label='n')
plt.plot(np.linspace(1,steps - 2, steps - 3), r_p_1[1:steps-2,1], label='k')
plt.title(label="Convergence of the parameters over " + str(steps) + " steps")
plt.legend()
plt.savefig('build/testing/convergence_test.pdf')
plt.close()

plt.figure
plt.plot(np.linspace(1,steps, steps - 1), delta_values[1:], label=r'$\delta$')
plt.title(label="Convergence of the parameters over " + str(steps) + " steps")
plt.legend()
plt.savefig('build/testing/delta_function.pdf')
plt.close()


"""Things that dont work:
    - Transferfunction gives weird values and is divergent for some inputs
    - From my understanding we try to find the postion where T_calc - T_measured = 0. But because this oscillates we use the delta function, which we then need to minimize.
        However the minimization seems to not work somehow. As its seems to be very dependent on the staring values, even though the paper suggests that its not
    - My minimizer is not minimizing? atleast the delta function is not getting smaller through iteration
        -> this seems to be the case when one of the values, either n or k passes 0    """