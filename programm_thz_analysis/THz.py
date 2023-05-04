import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import uncertainties as un
from matplotlib.cm import get_cmap
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c

#Read the excel file
data_ref = pd.read_excel('data/Unknown_Ref.xlsx')
data_sam = pd.read_excel('data/Unknown_Sample.xlsx')
data_ref = np.array(data_ref)
data_sam = np.array(data_sam)
data_ref[:,0] = data_ref[:,0] * 10**(-12)
data_sam[:,0] = data_sam[:,0] * 10**(-12)

#plot the intensity against time delay
plt.figure()
plt.plot(data_ref[:,0]*10**(12), data_ref[:,1], label='Reference')
plt.plot(data_sam[:,0]*10**(12), data_sam[:,1], label='Sample')
plt.xlabel(r'$ t/ps $')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.title('The reference and sample data set')
plt.savefig('THz1.pdf')
plt.close()

#############################################################


plt.figure()
plt.plot(data_ref[43:153, 0]*10**(12), data_ref[43:153,1], label='Reference Filtered') ## I have to find a way to find the post pulse with out hard coding
plt.plot(data_sam[80:190, 0]*10**(12), data_sam[80:190,1], label='Sample Filtered')
plt.xlabel(r'$ t/ps $')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.title('The filtered data sets')
plt.savefig('THz3_1.pdf')
plt.close()

ref_time = data_ref[:, 0]
sam_time = data_sam[:, 0]

data_ref = data_ref[43:153] #apply filter
data_sam = data_sam[80:190]

#########################################################################################
def FFT_func(I, t): 
    N = len(t) #length of t1
    timestep = np.abs(t[2]-t[3])
    FX = fft(I)[:N//2] #the fourier transform of the intensity. 
    FDelay = fftfreq(N, d=timestep)[:N//2] #FFT of the time to frequencies. 
    return [FDelay, FX]

freq_ref, amp_ref = FFT_func(data_ref[:,1], data_ref[:,0])  #in Hz
freq_sam, amp_sam = FFT_func(data_sam[:,1], data_sam[:,0])

plt.figure()
plt.plot(freq_ref* 10**(-12), np.abs(amp_ref), label='Reference filtered FFT')
plt.plot(freq_sam* 10**(-12), np.abs(amp_sam), label='Sample filtered FFT')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
plt.title('The FFT of the data sets without Post Pulse')
plt.savefig('THz4_1.pdf')
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_ref = freq_ref[1:] # we need to have same shapes
#somewhere here is the problem

H_0 = amp_sam/amp_ref
print(H_0)



angle = np.angle(H_0[1:]) #angle between complex numbers
phase = (np.unwrap(angle,period=np.pi))  #phase 


###########

d = 0.9*10**(-3) #millimeter

plt.figure()
#plt.plot(freq_ref*10**(-12),phase, label='phase')
plt.plot(freq_ref*10**(-12),angle, label='Phase')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\Phi$')
plt.xlim(0.18, 1)
plt.legend()
plt.grid()
plt.title('The phase')
plt.savefig('THzPhase.pdf')
plt.close()



n = 1 - c/(freq_ref* d) *phase
n_im = c/(freq_ref *d) *(np.log((4*n)/(n+1)**2)) - np.log(np.abs(phase))

plt.figure()
#plt.plot(freq_ref*10**(-12),phase, label='phase')
plt.plot(freq_ref*10**(-12),n_im, label='imagenary part of refractive index')
plt.plot(freq_ref*10**(-12), n, label='real part of refractive index')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 1)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index')
plt.savefig('THz5_1.pdf')
plt.close()


##################################################################################################
alpha = 2*freq_ref *n_im/c 

plt.figure()
plt.plot(freq_ref*10**(-12), alpha*10**(-2), label='Absorption coefficient')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 1)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient')
plt.savefig('THz6.pdf')
plt.close()