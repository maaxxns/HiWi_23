import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c
from scipy.optimize import curve_fit

def flatten(xss): # doesnt work for None type values
    return [x for xs in xss for x in xs]

def power(my_list):
    return [x**2 for x in my_list]

def gaussian(x, x_0, sigma = 0.1):
    N = len(x)
    return np.exp(-1/2*((x-x_0)/(N*sigma/2))**2)

def reverse_array(array):
    return array[::-1]

def H_0(data_ref, data_sam): #takes in two spectral amplitudes and dives them to return complex transfer function
    return (data_sam/data_ref)

def estimater_n(angle_T, omega, Material_parameter, substrate=None):
    if(substrate != None):
        return substrate - angle_T/( 2*np.pi*omega*Material_parameter.d/c)
    else: 
        return 1 - angle_T/( 2*np.pi*omega*Material_parameter.d/c)

def estimater_k(freq, T_meas, n_2, Material):
    A = (((n_2 - Material.n_1)*(n_2 - Material.n_3))/((n_2 + Material.n_1)*(n_2 + Material.n_3)) * np.cos(2*n_2 * freq * Material.d/c))
    D = ((n_2 + Material.n_1)*(n_2 + Material.n_3))/(2*n_2*(Material.n_1 + Material.n_3)) * np.exp(-Material.k_3*freq*Material.d/c)*T_meas
    k_2 = np.abs((c * np.log(np.abs(A))/np.log(np.abs(D))))/(4* 2*np.pi*freq*Material.d) 
    return k_2

def estimater_epsilon_1(n, k): # real part of epsilon dielectrical function
    return n**2 - k**2

def estimater_epsilon_2(n, k): # complex part of epsilon dielectrical function
    return 2*n*k

def estimater_sigma_1(freq, n, k):
    return (n*k*freq) #/2*np.pi

def estimater_sigma_2(freq, n, k):
    return (1 - (n**2 - k**2))*freq/2 #/(2*np.pi)

def absorption_coef(f, k): # takes complex refractice index at frequency f and returns the absorption coefficient at given frequency
    return 2*2*np.pi*f*k/(100*c)

def FFT_func(I, t): # FFT, I the Intensity of the signal as array of size X, t the time value of the signal of size X 
    N = len(t) #number of total data points
    timestep = np.abs(t[N//2 + 1]-t[N//2]) # the time between each data point
    FX = (fft(I)[:N//2]) #the fourier transform of the intensity. 
    FDelay = fftfreq(N, d=timestep)[:N//2] #FFT of the time to frequencies. 
    return [FDelay[10:], FX[10:]] # cut of the noise frequency

def difference_measured_calc(r, params):
    n = r[0]
    k = r[1]
    H_0_measured = params[0]
    phase_mes = params[1]
    freq = params[2]
    index = params[3]
    Material_parameter = params[4]
    FP = params[5]
    """ delta is the error between the estimated Transferfunction 
    and the measured transferfunction, so a delta of zero would mean we found the correct estimation 
    """ #if needed n_1, n_3 and k_1, k_3 can also be added 
    H_0_calc = Transfer_function_three_slabs(freq, n, k, Material_parameter, FP)
    phase_calc = (np.unwrap(np.angle(H_0_calc)))
    return np.abs(H_0_calc[index]-H_0_measured) + np.abs(phase_calc[index] - phase_mes)

def delta_of_r_whole_frequency_range(r, params):
    return delta_phi(r, params)**2 + delta_rho(r, params)**2 # this should be minimized in the process or best even be zero 

def delta_phi(r, params):
    n = r[0]
    k = r[1]
    H_0_measured = params[0]
    phase_mes = params[1]
    freq = params[2]
    index = params[3]
    Material_parameter = params[4]
    FP = params[5]
    H_0_calc = Transfer_function_three_slabs(freq, n, k, Material_parameter, FP)
    angle_0 = np.angle(H_0_calc) #angle between complex numbers
    phase_0 = np.unwrap(angle_0)[index]  #phase 
    return (phase_0 - phase_mes)

def delta_rho(r, params):
    n = r[0]
    k = r[1]
    H_0_measured = params[0]
    phase_mes = params[1]
    freq = params[2]
    index = params[3]
    Material_parameter = params[4]
    FP = params[5]
    H_0_calc = Transfer_function_three_slabs(freq, n, k, Material_parameter, FP)
    return (np.log(np.abs(H_0_calc[index])) - np.log(np.abs(H_0_measured)))
   
def Transfer_function_three_slabs(omega, n_2_real, k_2, Material_parameter, FP):
    n_1 = Material_parameter.n_1 - 1j*Material_parameter.k_1
    n_2 = n_2_real - 1j*k_2
    n_3 = Material_parameter.n_3 - 1j*Material_parameter.k_3
    T = (2*n_2*(n_1 + n_3)/((n_2 + n_1) * (n_2 + n_3))) * np.exp((-1j*n_2 + n_1*1j) * 2*np.pi*omega*Material_parameter.d/c)
    if(FP):
        FP = 1/(1 - (((n_2 - n_1)/(n_2 + n_1) * (n_2 - n_3)/(n_2 + n_3)) * np.exp(-2 * 1j*n_2 * 2*np.pi* omega*Material_parameter.d/c)))
        return T*FP
    else:
        return T
    
def Fabry_Perot(freq, r, Material): #calculates the FarbyPerot Factor for a given frequency
    n_1 = Material.n_1 - 1j*Material.k_1
    n_2 = r[0] - 1j*r[1] 
    n_3 = Material.n_3 - 1j*Material.k_3
    return 1/(1 - (((n_2 - n_1)/(n_2 + n_1) * (n_2 - n_3)/(n_2 + n_3)) * np.exp(-2 * 1j*n_2 * 2*np.pi*freq*Material.d/c)))# 

def inverse_Fabry_Perot(freq, r, Material): #calculates the FarbyPerot Factor for a given frequency
    n_1 = Material.n_1 - 1j*Material.k_1
    n_2 = r[0] - 1j*r[1] 
    n_3 = Material.n_3 - 1j*Material.k_3
    return (1 - (((n_2 - n_1)/(n_2 + n_1) * (n_2 - n_3)/(n_2 + n_3)) * np.exp(-2 * 1j*n_2 * 2*np.pi*freq*Material.d/c)))# 

def Transmissionfunction_curvefit(freq, H_0, p_0):
    params, cov = curve_fit(Transfer_function_three_slabs, freq, H_0, p_0)
    return params

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log(np.where(sd == 0, 0, m/sd))

def delta_delta_rho(kappa_2, index, params): # params: 0 = uncertainty T, 1 = T, 2 = kappa substrate, 3 = freq, 4 = uncertainty d 
    return (np.abs(params[0])/np.abs(params[1][index]) + np.abs(kappa_2 - params[2])*(params[3][index]/c) * params[4])

def delta_delta_phi(n_2, index, params): # params: 0 = uncertainty arg(T), 1 = n substrate, 2 = freq, 3 = uncertainty d
    return (params[0] + np.abs(n_2 - params[1])*(params[2][index]/c) * params[3])



##################################################################################################################################
# Zero_padding
##################################################################################################################################
#timestep = np.abs(data_ref[:,0][2]-data_ref[:,0][3]) # minimum time resolution
#N = len(data_ref[:,0]) #number of total data points
#
#num_zeros = 1500
#
#peak_ref,prop = find_peaks(data_ref[:,1], prominence=0.3) # finds the highest peak in the dataset and returns its index
#peak_ref = peak_ref[0:2]
#peak_sam,prop = find_peaks(data_sam[:,1], prominence=0.5)
#peak_sam = peak_sam[0:2]
#peak_sam[1] = peak_sam[1] - 100 # we assume that we cut off the array 50 steps before we hit the post pulse
#
#data_ref_zero = [np.append(data_ref[:peak_sam[1], 0], np.linspace(data_ref[peak_sam[1], 0], data_ref[peak_sam[1], 0]+num_zeros*timestep, num_zeros)),
#                 np.append(data_ref[:peak_sam[1], 1], (np.zeros(num_zeros)))]
#data_sam_zero = [np.append(data_sam[:peak_sam[1], 0], np.linspace(data_sam[peak_sam[1], 0], data_sam[peak_sam[1], 0]+num_zeros*timestep, num_zeros)),
#                 np.append(data_sam[:peak_sam[1], 1], (np.zeros(num_zeros)))]
#
##################################################################################################################################
# Some necessary calculations on frequency and time resolution
##################################################################################################################################

#Delta_f = 1/(N*timestep) #frequency resolution
#print('Delta t = ', " ",timestep/10**(-12), "ps")
#print("T ", N*timestep/10**(-12), "ps")
#print("Delta f = ", " ", Delta_f*10**(-12), "THz")


###################################################################################################################################
# This block applies the FFT to the zero padded data, aswell as masking frequencies that we dont need for the analization
###################################################################################################################################
#freq_ref_zero, amp_ref_zero = FFT_func(data_ref_zero[1], data_ref_zero[0])  #in Hz
#freq_sam_zero, amp_sam_zero = FFT_func(data_sam_zero[1], data_sam_zero[0])
#
#mask1_zero = freq_ref_zero < 4.5*10**12 # mask1_zero masks for THz frequency below 4.5 THz
#amp_ref_zero = amp_ref_zero[mask1_zero]
#amp_sam_zero = amp_sam_zero[mask1_zero]
#freq_ref_zero = freq_ref_zero[mask1_zero]
#freq_sam_zero = freq_sam_zero[mask1_zero]
