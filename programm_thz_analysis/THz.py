import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import uncertainties as un
from matplotlib.cm import get_cmap
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c

#Read the excel file
data_ref = np.genfromtxt('data/without_cryo_with_purge.txt', delimiter="	", comments="#")
data_sam = np.genfromtxt('data/without_cryo_with_purge_teflon_2.txt',  delimiter="	", comments="#")
data_ref[:,0] = data_ref[:,0] * 10**(-12) # ps in seconds
data_sam[:,0] = data_sam[:,0] * 10**(-12)

data_ref[:,0] = data_ref[:,0] + np.abs(np.min(data_ref[:,0])) # move everything to positiv times
data_sam[:,0] = data_sam[:,0] + np.abs(np.min(data_sam[:,0]))

#plot the intensity against time delay
plt.figure()
plt.plot(data_ref[:,0]*10**(12), data_ref[:,1], label='Reference')
plt.plot(data_sam[:,0]*10**(12) + 100, data_sam[:,1], label='Sample moved by +100ps')
plt.xlabel(r'$ t/ps $')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.title('The reference and sample data set')
plt.savefig('build/THz1.pdf')
plt.close()

#########################################################################################

def FFT_func(I, t): # FFT
    N = len(t) #number of total data points
    timestep = np.abs(t[2]-t[3]) # the time between each data point
    FX = fft(I)[:N//2] #the fourier transform of the intensity. 
    FDelay = fftfreq(N, d=timestep)[:N//2] #FFT of the time to frequencies. 
    return [FDelay[10:], FX[10:]] # cut of the noise frequency

freq_ref, amp_ref = FFT_func(data_ref[:,1], data_ref[:,0])  #in Hz
freq_sam, amp_sam = FFT_func(data_sam[:,1], data_sam[:,0])

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

freq_ref = freq_ref[1:] # we need to have same shapes
#somewhere here is the problem

H_0 = amp_sam/amp_ref # complex transfer function

angle = np.angle(H_0[1:]) #angle between complex numbers
phase = (np.unwrap(angle,period=2*np.pi))  #phase 


###########

d = 0.25*10**(-3) #millimeter

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


n = (1 - c/(freq_ref* d) *phase)

ln_a = np.log((4*n)/(n+1)**2)
ln_b = np.log(np.abs(phase))

n_im = c/(freq_ref *d) *(ln_a - ln_b)


plt.figure()
#plt.plot(freq_ref*10**(-12),phase, label='phase')
plt.plot(freq_ref*10**(-12),n_im, label='imagenary part of refractive index')
plt.plot(freq_ref*10**(-12), n, label='real part of refractive index')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
plt.ylim(-1,10)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index')
plt.savefig('build/THz5_1.pdf')
plt.close()


##################################################################################################
""""
alpha = 2*freq_ref *n_im/c 

plt.figure()
plt.plot(freq_ref*10**(-12), alpha, label='Absorption coefficient')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 1)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient')
plt.savefig('build/THz6.pdf')
plt.close()
"""