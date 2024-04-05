import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c
from scipy.optimize import curve_fit

def flatten(xss): # doesnt work for None type values
    return [x for xs in xss for x in xs]

def reverse_array(array):
    return array[::-1]

def lin(A, B, x):
    return A*x+B

def H_0(data_ref, data_sam): #takes in two spectral amplitudes and dives them to return complex transfer function
    return (data_sam/data_ref)

def n(freq, d, phase): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved phase and returns the real refractive index
    return (1 + (c* phase)/(freq* d) ) #    return (1 - c/(freq* d) *phase)

def estimater_n(angle_T, omega, Material_parameter, substrate=None):
    if(substrate != None):
        return substrate + angle_T/(omega*Material_parameter.d/c)
    else: 
        return 1 + angle_T/(omega*Material_parameter.d/c)

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
    FX = fft(I)[:N//2] #the fourier transform of the intensity. 
    FDelay = fftfreq(N, d=timestep)[:N//2] #FFT of the time to frequencies. 
    return [FDelay[10:], FX[10:]] # cut of the noise frequency

def error_function(H_0_calc, H_0_measured): # The error function needs to be minimized to find the values for n and k
    # a good explanation can be found  "A Reliable Method for Extraction of Material
    # Parameters in Terahertz Time-Domain Spectroscopy"
    delta_rho = np.zeros(H_0_calc.shape) # prepare arrays of same shape as H_0
    delta_phi = np.zeros(H_0_calc.shape)
    angle_mes = np.angle(H_0_measured)
    print(H_0_measured.size)
    if H_0_measured.size < 2:
        phase_mes = np.unwrap([angle_mes])
    else:
        phase_mes = np.unwrap(angle_mes)    
    angle_0 = np.zeros(H_0_calc.shape)
    phase_0 = np.zeros(H_0_calc.shape)
    for i in range(len(H_0_calc)):
        for j in range(len(H_0_calc)): # array should be symmetrical
            delta_rho[i,j] = np.log(np.abs(H_0_calc[i,j])) - np.log(np.abs(H_0_measured))
            angle_0[i,j] = np.angle(H_0_calc[i,j]) #angle between complex numbers
            phase_0[i,j] = np.unwrap([angle_0[i,j]])  #phase 
            delta_phi[i,j] = phase_0[i,j] - phase_mes
    print(delta_phi.shape, ' ', delta_rho.shape)
    return delta_phi**2 + delta_rho**2 # this should be minimized in the process

def delta_of_r(r, params):
    n = r[0]
    k = r[1]
    H_0_measured = params[0]
    phase_mes = params[1]
    freq = params[2]
    Material_parameter = params[3]
    """ delta is the error between the estimated Transferfunction 
    and the measured transferfunction, so a delta of zero would mean we found the correct estimation 
    """ #if needed n_1,n_3 and k_1, k_3 can also be added 
    H_0_calc = Transfer_function_three_slabs(freq, Material_parameter.n_1 ,n, Material_parameter.n_3, Material_parameter.k_1, k, Material_parameter.k_3, Material_parameter.d, True)
    #if(H_0_calc <= 0):
    #    print(H_0_calc, " n ", n, " k ", k)
    delta_rho = np.array([np.log(np.abs(H_0_calc[1])) - np.log(np.abs(H_0_measured))])[0]
    angle_0 = np.angle(H_0_calc) #angle between complex numbers
    #phase_0 = np.unwrap(angle_0)[1]  #phase 
    delta_phi = np.abs(angle_0 [1]- phase_mes)
    #print(delta_phi[0]**2 + delta_rho[0]**2)
    return delta_phi**2 + delta_rho**2 # this should be minimized in the process or best even be zero 

def delta_of_r_whole_frequency_range(r, params):
    n = r[0]
    k = r[1]
    H_0_measured = params[0]
    phase_mes = params[1]
    freq = params[2]
    index = params[3]
    Material_parameter = params[4]
    """ delta is the error between the estimated Transferfunction 
    and the measured transferfunction, so a delta of zero would mean we found the correct estimation 
    """ #if needed n_1,n_3 and k_1, k_3 can also be added 
    H_0_calc = Transfer_function_three_slabs(freq, Material_parameter.n_1 ,n, Material_parameter.n_3, Material_parameter.k_1, k, Material_parameter.k_3, Material_parameter.d, True)
    #if(H_0_calc <= 0):
    #    print(H_0_calc, " n ", n, " k ", k)
    delta_rho = np.array([np.log(np.abs(H_0_calc[index])) - np.log(np.abs(H_0_measured))])[0]
    angle_0 = np.angle(H_0_calc) #angle between complex numbers
    phase_0 = np.unwrap(angle_0)[index]  #phase 
    delta_phi = np.abs(phase_0- phase_mes)
    #print(delta_phi[0]**2 + delta_rho[0]**2)
    return delta_phi**2 + delta_rho**2 # this should be minimized in the process or best even be zero 


def Transfer_function(omega, n, k, l, fp):
    T = 4*n/(n + 1)**2 * np.exp(k*(omega*l/c)) * np.exp(-1j * (n - 1) *(omega*l/c))
    #if np.isinf(T) or np.isnan(T):
    #    print("Input values for T give rise to value error: n = ", n, " k = ", k, " omgea = ", omega)
    if(fp):
        FP = 1/(1 - ((1 - (n+1j*k))/(1 + (n + 1j*k)))**2 * np.exp(-2*1j * (n + 1j *k)* (omega*l/c)))
        return T*FP
    else:
        print("No FP factor")
        return T

def Transfer_function_three_slabs(omega, n_1_real, n_2_real, n_3_real, k_1, k_2, k_3, l, fp):
    n_1 = n_1_real + 1j*k_1
    n_2 = n_2_real + 1j*k_2 
    n_3 = n_3_real + 1j*k_3
    T = (2*n_2*(n_1 + n_3)/((n_2 + n_1) * (n_2 + n_3))) * np.exp(-(1j*n_2 - 1j*n_3) * omega*l/c)
    if(fp):
        FP = 1/(1 - (((n_2 - n_1)/(n_2 + n_1) * (n_2 - n_3)/(n_2 + n_3)) * np.exp(-2 * 1j*n_2 * omega*l/c)))
        return T*FP
    else:
        return T