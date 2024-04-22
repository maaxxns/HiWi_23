import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c
from scipy.optimize import curve_fit

def flatten(xss): # doesnt work for None type values
    return [x for xs in xss for x in xs]

def gaussian(x, x_0, sigma = 0.1):
    N = len(x)
    return np.exp(-1/2*((x-x_0)/(N*sigma/2))**2)

def reverse_array(array):
    return array[::-1]

def H_0(data_ref, data_sam): #takes in two spectral amplitudes and dives them to return complex transfer function
    return (data_sam/data_ref)

def n(freq, d, phase): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved phase and returns the real refractive index
    return (1 + (c* phase)/(freq* d) ) #    return (1 - c/(freq* d) *phase)

def estimater_n(angle_T, omega, Material_parameter, substrate=None):
    if(substrate != None):
        return substrate + angle_T/(omega*Material_parameter.d/c)
    else: 
        return 1 + angle_T/(omega*Material_parameter.d/c)

def estimater_k(freq, T_meas, n_2, Material):
    A = (((n_2 - Material.n_1)*(n_2 - Material.n_3))/((n_2 + Material.n_1)*(n_2 + Material.n_3)) * np.cos(2*n_2 * freq * Material.d/c))
    D = ((n_2 + Material.n_1)*(n_2 + Material.n_3))/(2*n_2*(Material.n_1 + Material.n_3)) * np.exp(-Material.k_3*freq*Material.d/c)*T_meas
    k_2 = np.abs((c * np.log(np.abs(A))/np.log(np.abs(D))))/(4*freq*Material.d) 
    return k_2

def absorption_coef(f, k): # takes complex refractice index at frequency f and returns the absorption coefficient at given frequency
    return 2*f*k/(100*c)

def k(freq, d, H_0, n): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved H_0 and returns the complex refractive index
    n_real = n
    ln_a = np.log((4*n_real)/(n_real + 1)**2)
    ln_b = np.log(np.abs(H_0))
    n_im = c/(freq *d) *(ln_a - ln_b)
    return n_im

def FFT_func(I, t): # FFT, I the Intensity of the signal as array of size X, t the time value of the signal of size X 
    N = len(t) #number of total data points
    timestep = np.abs(t[2]-t[3]) # the time between each data point
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
    H_0_calc = Transfer_function_three_slabs(freq, Material_parameter.n_1 ,n, Material_parameter.n_3, Material_parameter.k_1, k, Material_parameter.k_3, Material_parameter.d, FP)
    phase_calc = np.abs(np.unwrap(np.angle(H_0_calc)))
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
    H_0_calc = Transfer_function_three_slabs(freq, Material_parameter.n_1 ,n, Material_parameter.n_3, Material_parameter.k_1, k, Material_parameter.k_3, Material_parameter.d, FP)
    angle_0 = np.angle(H_0_calc) #angle between complex numbers
    phase_0 = np.unwrap(angle_0)[index]  #phase 
    return phase_0 - phase_mes

def delta_rho(r, params):
    n = r[0]
    k = r[1]
    H_0_measured = params[0]
    phase_mes = params[1]
    freq = params[2]
    index = params[3]
    Material_parameter = params[4]
    FP = params[5]
    H_0_calc = Transfer_function_three_slabs(freq, Material_parameter.n_1 ,n, Material_parameter.n_3, Material_parameter.k_1, k, Material_parameter.k_3, Material_parameter.d, FP)
    return np.array([np.log(np.abs(H_0_calc[index])) - np.log(np.abs(H_0_measured))])[0]
   
def Transfer_function_three_slabs(omega, n_1_real, n_2_real, n_3_real, k_1, k_2, k_3, l, fp):
    n_1 = n_1_real - 1j*k_1
    n_2 = n_2_real - 1j*k_2
    n_3 = n_3_real - 1j*k_3
    T = (2*n_2*(n_1 + n_3)/((n_2 + n_1) * (n_2 + n_3))) * np.exp(-(1j*n_2 - n_3*1j) * omega*l/c)
    if(fp):
        FP = 1/(1 - (((n_2 - n_1)/(n_2 + n_1) * (n_2 - n_3)/(n_2 + n_3)) * np.exp(-2 * 1j*n_2 * omega*l/c)))
        return T*FP
    else:
        return T
    
def Fabry_Perot(freq, r, Material): #calculates the FarbyPerot Factor for a given frequency
    n_1 = Material.n_1 - 1j*Material.k_1
    n_2 = r[0] - 1j*r[1] 
    n_3 = Material.n_3 - 1j*Material.k_3
    return 1/(1 - (((n_2 - n_1)/(n_2 + n_1) * (n_2 - n_3)/(n_2 + n_3)) * np.exp(-2 * 1j*n_2 *freq*Material.d/c)))# 

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
