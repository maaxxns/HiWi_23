import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from mathTHz import gaussian, Transfer_function_three_slabs, grad_T
from mathTHz import delta_of_r_whole_frequency_range
from scipy.signal import butter,sosfiltfilt
from scipy.constants import c
def lin(A, B, x):
    return A*x+B

def filter_dataset(data_ref,data_sam, filter=None): 
    """
        takes the signal data and applies filters based on which filter is given.
        returns the filtered datasets.
        possible filters are gaussian and truncating
    """
    # Some test with filters for the dataset
    if(filter == "gaussian"): #puts a gaussian function ontop of the signal peak
        normalizer = np.amax(np.abs(data_ref[:,1]))
        data_ref[:,1] = data_ref[:,1]/normalizer
        data_sam[:,1] = data_sam[:,1]/normalizer
        x = np.linspace(0,len(data_ref[:,0]),len(data_ref[:,0]))
        peak_sam = np.argmax(data_sam[:,1])
        peak_ref = np.argmax(data_ref[:,1])
        data_sam[:,1] = data_sam[:,1]*gaussian(x, peak_sam, sigma=0.05) # dataset with gaussian filter
        data_ref[:,1] = data_ref[:,1]*gaussian(x, peak_ref, sigma=0.05)
    if(filter == "truncate"): # truncates the signal to just include the first peak and some signal after it but not the post pulse
        #peak_sam,prop = find_peaks(data_sam[:,1], height=(None, np.argmax(data_sam[:,1])/1.2))# finds the highest peak in the dataset and returns its index
        #peak_ref,prop = find_peaks(data_ref[:,1], height=(None, np.argmax(data_ref[:,1])/1.2)) # finds the highest peak in the dataset and returns its index
        #second_highest_peak_index_sam = peak_sam[np.argpartition(prop['peak_heights'],-2)[-1]]
        #second_highest_peak_index_ref = peak_ref[np.argpartition(prop['peak_heights'],-2)[-1]]
        peak_ref = np.argmax(data_ref[:,1])
        peak_sam = np.argmax(data_sam[:,1])
        t = np.abs(data_ref[10,0] - data_ref[11,0])
        width_peak = int((4.5*10**(-12))/t)
        cut_ref = peak_ref + width_peak #- len(data_sam)//1 # we move a bit to the left from the second peak, so that we dont include the peak. In this case we move a tenth of the whole signal length
        cut_sam = peak_sam + width_peak #- len(data_ref)//1 # we move a bit to the left from the second peak, so that we dont include the peak. In this case we move a tenth of the whole signal length
        data_ref[cut_ref:,1] = np.zeros(len(data_ref[cut_ref:,1])) #truncate the signal according to the second peak in the sample data, which should be the post pulse. And substitute the values with zeros
        data_sam[cut_sam:,1] = np.zeros(len(data_sam[cut_sam:,1])) #truncate the signal according to the second peak in the sample data, which should be the post pulse
         #truncate the sample data aswell
    return data_ref, data_sam

def grad_2D(func, r, params=None, h = 10**(-6)): 
    grad_0_x = (func([r[0] + h, r[1]], params) - func([r[0] - h, r[1]], params))/2*h
    grad_0_y = (func([r[0],r[1] + h], params) - func([r[0], r[1] - h], params))/2*h 
    return np.array([grad_0_x, grad_0_y])

def grad_of_delta(r, params):
    H_0_measured = params[0]
    phase_mes = params[1]
    freq = params[2]
    index = params[3]
    Material_parameter = params[4]
    FP = params[5]
    A = np.array([(2*(r[0] - Material_parameter.n_1)*(freq*Material_parameter.d/c)**2)[index], 0])
    T_0 = Transfer_function_three_slabs(freq, r[0], r[1], Material_parameter, FP)[index]
    gr = grad_T(freq[index], r[0], r[1], Material_parameter, FP)
    B = (2*((np.log(np.abs(T_0))) - np.log(np.abs(H_0_measured))))* grad_T(freq[index], r[0], r[1], Material_parameter, FP)/T_0
    grad = A + B
    return np.array(A + B)

def grad_2D_minizer(r, params=None, h =0.0065): 
    func = delta_of_r_whole_frequency_range
    grad_0_x = (func([r[0] + h, r[1]], params) - func([r[0] - h, r[1]], params))/2*h
    grad_0_y = (func([r[0],r[1] + h], params) - func([r[0], r[1] - h], params))/2*h 
    return np.array([grad_0_x, grad_0_y])


def Hessematrix(func, r, params=None, h = 10**(-6)):  
    A = (func([r[0] + h, r[1]], params) - 2*func([r[0], r[1]], params) + func([r[0] - h, r[1]], params))/h**2
    B = (func([r[0] + h/2, r[1] + h/2], params) - func([r[0] + h/2, r[1] - h/2], params) - func([r[0] - h/2, r[1] + h/2], params) + func([r[0] - h/2, r[1] - h/2], params))/h**2
    C = B # Satz von Schwarz
    D = (func([r[0], r[1] + h], params) - 2*func([r[0], r[1]], params) + func([r[0], r[1] - h], params))/h**2
    # A = d²delta(r_p)/dn² = (delta(n + h, k) - 2*delta(n,k) - delta(n-h,k))/h²
    # B = d²delta(r_p)/dkdn = (delta(n + h/2, k + h/2) - delta(n + h/2, k - h/2) - delta(n - h/2, k + h/2)  + delta(n - h/2, k - h/2))/h²
    # D = d²delta(r_p)/dk² = (delta(n, k + h) - 2*delta(n,k) - delta(n,k - h))/h²
    return np.array([[A,B], [C,D]])

def Hessematrix_minizer(r, params=None, h = 10**(-6)):  
    func = delta_of_r_whole_frequency_range
    A = (func([r[0] + h, r[1]], params) - 2*func([r[0], r[1]], params) + func([r[0] - h, r[1]], params))/h**2
    B = (func([r[0] + h/2, r[1] + h/2], params) - func([r[0] + h/2, r[1] - h/2], params) - func([r[0] - h/2, r[1] + h/2], params) + func([r[0] - h/2, r[1] - h/2], params))/h**2
    C = B # Satz von Schwarz
    D = (func([r[0], r[1] + h], params) - 2*func([r[0], r[1]], params) + func([r[0], r[1] - h], params))/h**2
    # A = d²delta(r_p)/dn² = (delta(n + h, k) - 2*delta(n,k) - delta(n-h,k))/h²
    # B = d²delta(r_p)/dkdn = (delta(n + h/2, k + h/2) - delta(n + h/2, k - h/2) - delta(n - h/2, k + h/2)  + delta(n - h/2, k - h/2))/h²
    # D = d²delta(r_p)/dk² = (delta(n, k + h) - 2*delta(n,k) - delta(n,k - h))/h²
    return np.array([[A,B], [C,D]])

def newton_minimizer(func, r, params, h=10**(-3), gamma = 1): #newton iteration step to find the best value of r=(n_2,k_2)  
    A = Hessematrix(func, r, params, h) # Calculate the hessian matrix of delta(r_p) 
    grad_ = grad_2D(func,r, params, h) # calculate the gradient of delta(r_p)
    r_p_1 = r - gamma * np.linalg.inv(A).dot(grad_) #why is r_p going in negativ direction when both the hesse and the gradient are negativ, should the r_p move in positiv direction than?
    # r_p+1 = r_p - A⁽⁻¹⁾*grad(delta(r_p))
    return r_p_1 # returns new values for [n_2,k_2] that minimize the error according to newton iteration step 

def gradient_decent(func, r, params, h = 10**-6, gamma = 1):
    grad_ = grad_2D(func, r, params, h)
    r_p_1 = r - gamma*grad_
    return r_p_1

def linear_approx(x, y): # Fits a linear function into the data set where x is usually the frequency and y is the phase. But could also be used for any x=arraylike y=arraylike
    boundaries = len(x)//2
    upper_bound = boundaries + int(boundaries/3)
    #lower_bound = boundaries - int(boundaries/4)
    lower_bound = 0
    params, cov = curve_fit(lin, x[lower_bound:upper_bound], y[lower_bound:upper_bound])
    return params

def Transmission_approx(freq, H_0, r):
    params, cov = curve_fit(Transfer_function_three_slabs, freq, H_0, r)
    return params

def lowpass(data: np.ndarray, cutoff: float, poles: int = 5):
    sos = butter(poles, cutoff, 'lowpass', output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data