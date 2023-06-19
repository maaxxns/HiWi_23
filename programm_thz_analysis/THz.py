import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c
from scipy.optimize import curve_fit
###################################################################################################################################
# Functions

def lin(A, B, x):
    return A*x+B

def H_0(data_ref, data_sam): #takes in two spectral amplitudes and dives them to return complex transfer function
    return (data_sam/data_ref)

def n(freq, d, phase): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved phase and returns the real refractive index
    return (1 + (c* phase)/(freq* d) ) #    return (1 - c/(freq* d) *phase)


def k(freq, d, H_0, n): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved H_0 and returns the complex refractive index
    n_real = n
    ln_a = np.log((4*n_real)/(n_real + 1)**2)
    ln_b = np.log(np.abs(H_0))
    n_im = c/(freq *d) *(ln_a - ln_b)
    return n_im


def FFT_func(I, t): # FFT
    N = len(t) #number of total data points
    timestep = np.abs(t[2]-t[3]) # the time between each data point
    FX = fft(I)[:N//2] #the fourier transform of the intensity. 
    FDelay = fftfreq(N, d=timestep)[:N//2] #FFT of the time to frequencies. 
    return [FDelay[10:], FX[10:]] # cut of the noise frequency

def unwrapping_alt(amp_ref, amp_sam, freq_ref): #unwrapping from paper Phase_correction_in_THz_TDS_JIMT_revision_clean.pdf
    angle_ref = np.angle(amp_ref)
    angle_sam = np.angle(amp_sam)
    phase_dif = (np.unwrap(abs(angle_ref - angle_sam)))
    params,_ = curve_fit(lin, freq_ref, phase_dif)
    phase_dif_0 = phase_dif - 2*np.pi*np.floor(params[1]/np.pi)
    phase_ref_0 = freq_ref * t_peak_ref
    phase_sam_0 = freq_sam * t_peak_sam
    phase_offset = freq_ref * (t_peak_sam - t_peak_ref)
    return phase_dif_0 - phase_ref_0 + phase_sam_0 + phase_offset

 

###################################################################################################################################
#       All data is read in, in this block.
#       Necessary data are
#       - the time resolved THz measurment
#       - the thickness of the probe
###################################################################################################################################


# The thickness of the probe

d = 0.380*10**(-3) # thickness of the probe in SI

#Read the excel file

data_sam = np.genfromtxt('data/15_06_THz_measurment/Si_wafer.txt', delimiter="	", comments="#") # The time resolved dataset of the probe measurment
data_ref = np.genfromtxt('data/15_06_THz_measurment/empty.txt',  delimiter="	", comments="#") # the time resolved dataset of the reference measurment


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
plt.title('The reference and sample data set')
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
peak_sam,prop = find_peaks(data_sam[:,1], prominence=1)
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
plt.title('The reference and sample data set with zero padding')
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
plt.title('The FFT of the data sets')
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
plt.title('The FFT of the zero padded data sets')
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
plt.plot(freq_ref*10**(-12),phase, label='phase directly from H_0')
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
plt.title('The phase unwarpped of the zero padded data')
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
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,4)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index')
plt.savefig('build/THz_real_index.pdf')
plt.close()

###################################################################################################################################
#   This block plots the real part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), n(freq_ref, d, phase_dif), label='real part of refractive index calculated by the alternative unwrap')
plt.plot(freq_ref_zero*10**(-12), n_real_zero, label='real part of refractive index')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,4)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index from zero padded data')
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
plt.title('The complex part of the refractive index')
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
plt.title('The complex part of the refractive index from zero padded data')
plt.savefig('build/THz_complex_index_zero.pdf')
plt.close()

###################################################################################################################################
# This block calculates the absorption coefficient and plots it
###################################################################################################################################

alpha = 2*freq_ref *n_im/c 

plt.figure()
plt.plot(freq_ref*10**(-12), alpha/100, label='Absorption coefficient')
#plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], label='refference n from steffens program')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 4.5)
plt.ylim(0, 500)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient')
plt.savefig('build/THz_absorption.pdf')
plt.close()
