import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *
from plot import *
from tqdm import tqdm
###################################################################################################################################
# Datatypes

@dataclass
class Material_parameters:
    d: float
    n_1: float
    k_1: float 
    n_3: float 
    k_3: float

"""
    The algorythm is very unstable depeding on what the starting parameter is.
    For the test data set in which I use "smooth" data this is not the case.
    I Already employ an estimator for the real part of the refractive index, but not for the complex part.
    Even with the estimator the estimation process for other values other than the intial one is quite hard to find at some frequencies.
    Depeding on the epsilon (break condition from step one to step two has to be smaller than epsilon) and the stepsize the apgorythm takes between 15-50 minutes for 1000 frequency values.
    Usually I run the algo at epsilon = 10**-4 - 10**-3 and h = 0.05 - 0.06. This results in a time of about 20 minutes for 1000 steps but should decreas drastically if the intial guess would be better.

    Idea:
        Better preprocessing.
            Usually I give the algo all the data in time domain and filter out too high and too low frequencies afterwards.
            But I think I should try to give the algorythm just a specific timeframe, so that the pre and post pulse will not be inlcuded in the data

    Eliminate Oscillations:
        - Divide by FP
        - Measure over bigger intervall

    """


###################################################################################################################################
###################################################################################################################################
#       All data is read in, in this block.
#       Necessary data are
#       - the time resolved THz measurment
#       - the thickness of the probe
###################################################################################################################################


# The thickness of the probe

d = 2070*10**(-6) # thickness of the probe in SI
n_air = 1
n_slab = 1
k_slab = 1

Material = Material_parameters(d = d, n_1=n_air, k_1=n_air, n_3=n_slab, k_3=k_slab)

#Read the excel file
data_sam = np.genfromtxt('data/20240409/HDPE_2070um.txt', delimiter="	", comments="#") # The time resolved dataset of the probe measurment
data_ref = np.genfromtxt('data/20240409/reference.txt',  delimiter="	", comments="#") # the time resolved dataset of the reference measurment

###################################################################################################################################

data_ref[:,0] = data_ref[:,0] * 10**(-12) # ps in seconds
data_sam[:,0] = data_sam[:,0] * 10**(-12)

data_ref[:,0] = data_ref[:,0] + np.abs(np.min(data_ref[:,0])) # move everything to positiv times
data_sam[:,0] = data_sam[:,0] + np.abs(np.min(data_sam[:,0]))

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

##################################################################################################################################
# Some necessary calculations on frequency and time resolution
##################################################################################################################################

Delta_f = 1/(N*timestep) #frequency resolution
print('Delta t = ', " ",timestep/10**(-12), "ps")
print("T ", N*timestep/10**(-12), "ps")
print("Delta f = ", " ", Delta_f*10**(-12), "THz")

###################################################################################################################################
#           Filters if wanted  
###################################################################################################################################
filter_0 = False
if(filter_0):
    x = np.linspace(0,len(data_sam[:,0]),len(data_sam[:,0]))
    plot_gaussian(data_sam[:,0], gaussian(x, find_peaks(data_sam[:,1], prominence=1)[0][0]), data_sam)
    data_sam = filter_dataset(data_sam)
    data_ref = filter_dataset(data_ref)
    print("Gaussian function used as filter for preprocessing")
else:
    print("No preprocessing filter")

###################################################################################################################################
# This block applies the FFT to the data, aswell as masking frequencies that we dont need for the analization
###################################################################################################################################

freq_ref, amp_ref = FFT_func(data_ref[:,1], data_ref[:,0])  #in Hz
freq_sam, amp_sam = FFT_func(data_sam[:,1], data_sam[:,0])

mask1 = freq_ref < 3*10**12 # mask1ed for THz frequency below 4.5 THz
amp_ref = amp_ref[mask1]
amp_sam = amp_sam[mask1]
freq_ref = freq_ref[mask1]
freq_sam = freq_sam[mask1]
mask2 = 200*10**9 < freq_ref # mask1ed for THz frequency above 200 GHz 
amp_ref = amp_ref[mask2]
amp_sam = amp_sam[mask2]
freq_ref = freq_ref[mask2]
freq_sam = freq_sam[mask2]


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

###################################################################################################################################
# This block calculates the complex transfer function and does the unwrapping porcess
###################################################################################################################################

H_0_value = H_0(amp_ref, amp_sam) # complex transfer function

angle = np.angle(H_0_value) #angle between complex numbers
phase = np.unwrap(angle)  #phase 

###################################################################################################################################
# This block calculates the complex transfer function and does the unwrapping porcess
###################################################################################################################################

H_0_value_zero = H_0(amp_ref_zero, amp_sam_zero) # complex transfer function

angle_zero = np.angle(H_0_value_zero) #angle between complex numbers
phase_zero = np.unwrap(angle_zero)  #phase 
phase_approx  = linear_approx(freq_ref, phase)[1] * freq_ref - linear_approx(freq_ref, phase)[0]

###################################################################################################################################
#   This block calculates the real and complex part of the refractive index
###################################################################################################################################

n_real = n(freq_ref, d, phase)
n_im = k(freq_ref, d, H_0_value, n_real)

###################################################################################################################################
#   This block calculates the real and complex part of the refractive index
###################################################################################################################################

n_real_zero = n(freq_ref_zero, d, phase_zero)
n_im_zero = k(freq_ref_zero, d, H_0_value_zero, n_real_zero)

###################################################################################################################################
# This block calculates the absorption coefficient and plots it
###################################################################################################################################

alpha = 2*freq_ref *n_im/c 

###################################################################################################################################
# This block calculates the absorption coefficient for zero padding and plots it
###################################################################################################################################

alpha_zero = 2*freq_ref_zero*n_im_zero/c

###################################################################################################################################
#       testing stuff out 
###################################################################################################################################
"""For thin samples the Transmission function oscillates because the Thz is reflected inside the sample multiple times.
Lets see what the frequency of the osciallation is"""
t_T_real, amp_T_real = FFT_func(H_0_value.real, freq_ref)
t_T_imag, amp_T_imag = FFT_func(H_0_value.imag, freq_ref)

plt.figure()
plt.plot(t_T_real*10**9, amp_T_real, label='real')
plt.plot(t_T_imag*10**9, amp_T_imag, label='imag')
plt.xlabel(r'$ t / ns $')
plt.ylabel(r'$Fabry Perot$')
#plt.ylim(0, 500)
plt.legend()
plt.grid()
plt.title('FabryPerot in t i guess')
plt.savefig('build/FabryPerot_FFT.pdf')
plt.close()
###################################################################################################################################
#       All the plotting
###################################################################################################################################

plotting = True
if(plotting):
    print("Plotting...\n")
    plot_Intensity_against_time(data_ref, data_sam)
    plot_Intensity_against_time_zeropadding(data_ref_zero, data_sam_zero)
    plot_FFT(freq_ref, amp_ref, freq_ref, amp_sam)
    plot_FFT(freq_ref_zero, amp_ref_zero, freq_sam_zero, amp_sam_zero, zeropadded=True)
    plot_phase_against_freq(freq_ref, phase, angle)
    plot_phase_against_freq(freq_ref_zero, phase_zero, angle_zero, zeropadded=True)
    plot_phase_against_freq(freq_ref, phase, angle, zeropadded=False, approx=True, phase_approx=phase_approx)
    plot_realpart_refractive_index(freq_ref, n_real)
    plot_realpart_refractive_index(freq_ref_zero, n_real_zero, zeropadded=True)
    plot_complex_refrective_index(freq_ref, n_im)
    plot_complex_refrective_index(freq_ref_zero, n_im_zero, zeropadded=True)
    plot_absorption_coefficient(freq_ref, alpha)
    plot_absorption_coefficient(freq_ref_zero, alpha_zero, zeropadded=True)
    plot_H_0_against_freq(freq_ref, (H_0_value))
    plot_H_0_against_freq(freq_ref_zero, H_0_value_zero, True)
    plot_FabryPerot(freq_ref, Fabry_Perot(freq_ref, [3.4,3.4], Material))
###################################################################################################################################
# Here Starts the numerical process of finding the refractive index
###################################################################################################################################

###################################################################################################################################
print("-----------------------------------------------------------")

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
    # Minimizing the delta function with newton 
###################################################################################################################################

###################################################################################################################################
# Set starting values for the algorythm
###################################################################################################################################
minlimit = 1 # lower limit of the frequency range. 1 for no lower limit. optimal 200 GHz
maxlimit = -2 # upper limit of the frequency range. -2 for no upper limit. optimal 5THz

n_0 = estimater_n(angle, freq_ref, Material, substrate=1)[maxlimit] #intial guess for n, calculated as in the paper "A Reliable Method for Extraction of Material
                                                       #Parameters in Terahertz Time-Domain Spectroscopy
                                                       #Lionel Duvillaret, Frédéric Garet, and Jean-Louis Coutaz"
k_0 = n_0 # initial guess for k
h = 0.006 #step size for Newton or rather the gradient/hessian matrix
num_steps = 50000 # maximum number of steps taken per frequency if the break condition is not
freq_min_lim = 1*10**12 #lower frequency limit # to be implemnted
freq_max_lim = 3*10**12 #upper frequency limit
epsilon = 10**-4

###################################################################################################################################
steps = np.linspace(1, num_steps, num_steps, dtype=int)
r_0 = np.array([n_0,k_0]) # r_p[0] = n, r_p[1] = k 
r_per_step = [None]*(len(steps) + 1)
r_per_freq = [None]*len(freq_ref) # all the n and k per frequency will be written into this array

r_per_step[0] = r_0

print("starting values for Newton-Raphson: r_per_step =", r_0, ", h = ", h)
threshold_n = 0.9
threshold_k = 0.1
kicker_n, kicker_k = 0.5, 0.5


if(Material.d < 10**-3):
    """
    If the material can be considered optically thick we can discard the Fabry Perot factor as reflections inside the material will not travel from on side to the other fast enough.
    However, if the material is thin we have to take the FP into account which means that we have to find the right value for FP, n and k.
    For this the idea is:
        1. estimate n and k roughly (should already be done with the estimator function)
        2. estimate a FP value 
        3. Divide H_0 (The transmission function) by FP
        4. Do the same process as if the material would be thick and find a n and k
        5. Start the process over until a good value for n and k is found
    """
    FP = False
    for freq in tqdm(reverse_array(freq_ref[minlimit:maxlimit])): #walk through frequency range from upper to lower limit
        index = np.argwhere(freq_ref==freq)[0][0]
        params_delta_function = [H_0_value[index], phase_approx[index], freq_ref, index, Material, FP]
        for step in steps:
            Fabry_Perot_value = Fabry_Perot(freq_ref, r_per_step[step - 1], Material)
            params_delta_function[0] = params_delta_function[0]/Fabry_Perot_value[index] # Divide the H_0_value by the estimated FP factor to get rid of the oscialltions
            params_delta_function[1] = params_delta_function[1] - np.unwrap(np.angle(Fabry_Perot_value))[index] # Not sure about this one, but it should be necessary to also clean the phase from the Fabry Perot factor and because phase value need to be substracted if the complex number is divide it should work like this
            r_per_step[step] = newton_minimizer(delta_of_r_whole_frequency_range, r_per_step[step - 1], params=params_delta_function, h = h)
            r_0 = r_per_step[step]
            params_delta_function = [H_0_value[index], phase_approx[index], freq_ref, index, Material, FP]
            print(delta_of_r_whole_frequency_range(r_per_step[step - 1], params=params_delta_function))
            if(delta_of_r_whole_frequency_range(r_per_step[step - 1], params=params_delta_function) < 10): #break condition for when the values seems to be fine
                break
            if(r_per_step[step][0] < threshold_n): # This is just a savety measure if the initial guess is not good enough
                r_per_step[step][0] = r_0[0] + kicker_n
                kicker_n = kicker_n + 0.5
                #print("kicker used for n, kicker at: ", kicker_n)
                if(step > 1000):
                    step = step - 1000 # every time we engage the kicker we give the algo a bit time
            if(r_per_step[step][1] < threshold_k):
                r_per_step[step][1] = r_0[1] + kicker_k
                kicker_k = kicker_k + 0.5
                #print("kicker used for k, kicker at: ", kicker_k)
                if(step > 1000):
                    step = step - 1000 # every time we engage the kicker we give the algo a bit time
            if(delta_of_r_whole_frequency_range(r_per_step[step - 1], params=params_delta_function) - delta_of_r_whole_frequency_range(r_per_step[step], params=params_delta_function) < 0.1):
                r_per_step[step][0] = r_0[0] + kicker_n
                r_per_step[step][1] = r_0[1] + kicker_k
                kicker_n = kicker_n - 0.2
                kicker_k = kicker_k - 0.2
                #print("kicker used for k, kicker at: ", kicker_k)
                if(step > 1000):
                    step = step - 1000 # every time we engage the kicker we give the algo a bit time
        kicker_n, kicker_k = 0.5, 0.5 # reset kickers
        r_per_freq[index] = [r_0[0], r_0[1]] # save the final result of the Newton method for the frequency freq
        r_per_step[0] = r_0 # use the n and k value from the last frequency step as guess for the next frequency
else:
    FP = True
    for freq in tqdm(reverse_array(freq_ref[minlimit:maxlimit])): #walk through frequency range from upper to lower limit
        index = np.argwhere(freq_ref==freq)[0][0]
        params_delta_function = [H_0_value[index], phase_approx[index], freq_ref, index, Material, FP]
        for step in steps:
            r_per_step[step] = newton_minimizer(delta_of_r_whole_frequency_range, r_per_step[step - 1], params=params_delta_function, h = h)
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


#r_per_freq = reverse_array(r_per_freq) # we need to turn the array back around

alpha = absorption_coef(freq_ref[minlimit:maxlimit], flatten(r_per_freq[minlimit:maxlimit])[1::2])

print("Done")

print("Plotting...")
plt.figure()
plt.plot(freq_ref[minlimit:maxlimit]/1e12, flatten(r_per_freq[minlimit:maxlimit])[0::2], label='n') # we have to flatten the array before it plots 
plt.plot(freq_ref[minlimit:maxlimit]/1e12, flatten(r_per_freq[minlimit:maxlimit])[1::2], label='k')
#plt.plot(freq_ref[minlimit:maxlimit]/1e12, alpha, label=r'$\alpha$')

plt.xlabel(r'$\omega / THz$')
plt.ylabel('value')
plt.title('parameter: epsilon ' + str(epsilon) + ', h ' + str(h) + ', kicker n\n' + str(kicker_n) + ', kicker k' + str(kicker_k) + ', start r ' + str([n_0, k_0]))
plt.legend()
plt.savefig('build/testing/frequncy_against_n_k.pdf')
