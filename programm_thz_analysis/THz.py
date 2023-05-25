import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import uncertainties as un
from matplotlib.cm import get_cmap
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c

###################################################################################################################################
# Functions

def H_0(data_ref, data_sam): #takes in two spectral amplitudes and dives them to return transfer function
    return (data_sam/data_ref)

def n(freq, d, phase): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved phase and returns the real refractive index
    return (1 - c/(freq* d) *phase)

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

###################################################################################################################################

#Read the excel file
data_sam = np.genfromtxt('data/without_cryo_with_purge_teflon.txt', delimiter="	", comments="#")
data_ref = np.genfromtxt('data/without_cryo_with_purge_teflon_2.txt',  delimiter="	", comments="#")


data_ref[:,0] = data_ref[:,0] * 10**(-12) # ps in seconds
data_sam[:,0] = data_sam[:,0] * 10**(-12)

data_ref[:,0] = data_ref[:,0] + np.abs(np.min(data_ref[:,0])) # move everything to positiv times
data_sam[:,0] = data_sam[:,0] + np.abs(np.min(data_sam[:,0]))

peaks_inref,_ = find_peaks(data_ref[:,1], prominence = 0.3) # height=np.max(data_ref[:,1])
peaks_insam,_ = find_peaks(data_sam[:,1], prominence = 0.3)

cap_index_ref = abs(peaks_inref[0] - peaks_inref[-1])//2 + peaks_inref[0]
cap_index_sam = abs(peaks_insam[0] - peaks_insam[-1])//2 + peaks_insam[0]

#data_ref = data_ref[:cap_index_ref]
#data_sam = data_sam[:cap_index_sam]

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
plt.savefig('build/THz1.pdf')
plt.close()

#########################################################################################

freq_ref, amp_ref = FFT_func(data_ref[:,1], data_ref[:,0])  #in Hz
freq_sam, amp_sam = FFT_func(data_sam[:,1], data_sam[:,0])

mask1 = freq_ref < 4.5*10**12 # mask1ed for THz frequency below 4.5 THz
amp_ref = amp_ref[mask1]
amp_sam = amp_sam[mask1]
freq_ref = freq_ref[mask1]
freq_sam = freq_sam[mask1]

##########################################################################################
# read in data that is already in frequency domain

data_in_freq_domain = np.genfromtxt('data/teflon_1/teflon_1_frequency_domain.txt', delimiter="	", comments="#")[10:] 
# first row is freq, second sample amplitude, fourth reffernce 


material_properties_ref = np.genfromtxt('data/teflon_1/teflon_1_material_properties.txt', delimiter="	", comments="#")[10:]

mask2 = data_in_freq_domain[:,0] < 4.5
data_in_freq_domain = data_in_freq_domain[mask2]

plt.figure()
plt.plot(data_in_freq_domain[:,0], data_in_freq_domain[:,1], label='Sample FFT') # plot in Thz
plt.plot(data_in_freq_domain[:,0], data_in_freq_domain[:,3], label='Reference FFT') # plot in Thz
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
plt.title('The FFT of the data sets')
plt.savefig('build/THz4_2.pdf')
plt.close()

##########################################################################################

plt.figure()
plt.plot(freq_ref* 10**(-12), np.abs(amp_ref), label='Reference FFT') # plot in Thz
plt.plot(freq_sam* 10**(-12), np.abs(amp_sam), label='Sample FFT')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
plt.title('The FFT of the data sets')
plt.savefig('build/THz4_1.pdf')
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_ref = freq_ref # we need to have same shapes

H_0_value = H_0(amp_ref, amp_sam) # complex transfer function

angle = np.angle(H_0_value) #angle between complex numbers
phase = (np.unwrap(angle))  #phase 


plt.figure()
plt.plot(freq_ref*10**(-12),angle, label='angle')
plt.plot(freq_ref*10**(-12),phase, label='Phase')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\Phi$')
plt.legend()
plt.grid()
plt.title('The phase and the complex angle. Note the angle is unwrapped.')
plt.savefig('build/THzPhase.pdf')
plt.close()


d = 0.26*10**(-3) #millimeter
n_real = n(freq_ref, d, phase)
n_im = k(freq_ref, d, H_0_value, n_real)


plt.figure()
#plt.plot(freq_ref*10**(-12),phase, label='phase')
plt.plot(freq_ref*10**(-12), n_real, label='real part of refractive index')
plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,1], label='refference n from steffens program')
#plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], label='refference k from steffens program')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
plt.ylim(0,2)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index')
plt.savefig('build/THz5_1.pdf')
plt.close()


##################################################################################################

plt.figure()
plt.plot(freq_ref*10**(-12), n_im, label='complex part of refractive index')
#plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], label='refference n from steffens program')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('k (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,2000)
plt.legend()
plt.grid()
plt.title('The complex part of the refractive index')
plt.savefig('build/THz5_2.pdf')
plt.close()

alpha = 2*freq_ref *n_im/c 

plt.figure()
plt.plot(freq_ref*10**(-12), alpha/100, label='Absorption coefficient')
plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], label='refference n from steffens program')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 4.5)
plt.ylim(0, 500)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient')
plt.savefig('build/THz6.pdf')
plt.close()
