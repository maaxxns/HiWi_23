import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from mathTHz import gaussian, Transfer_function_three_slabs
from mathTHz import delta_of_r_whole_frequency_range
def lin(A, B, x):
    return A*x+B

def filter_dataset(data):
    peak,prop = find_peaks(data[:,1], prominence=1) # finds the highest peak in the dataset and returns its index
    peak = peak[0]
    x = np.linspace(0,len(data[:,0]),len(data[:,0]))
    # Some test with filters for the dataset
    data[:,1] = data[:,1]#/np.amax(np.abs(data[:,1]))
    data[:,1] = data[:,1]*gaussian(x, peak, sigma=0.05) # dataset with gaussian filter
    return data

def grad_2D(func, r, params=None, h = 10**(-6)): 
    grad_0_x = (func([r[0] + h, r[1]], params) - func([r[0] - h, r[1]], params))/2*h
    grad_0_y = (func([r[0],r[1] + h], params) - func([r[0], r[1] - h], params))/2*h 
    return np.array([grad_0_x, grad_0_y])

def grad_2D_minizer(r, params=None, h = 10**(-6)): 
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


def newton_minimizer(func, r, params, h=10**(-6), gamma = 1): #newton iteration step to find the best value of r=(n_2,k_2)  
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
    upper_bound = boundaries + boundaries//2
    lower_bound = boundaries - boundaries//2
    upper_bound = lower_bound
    lower_bound = 1
    params, cov = curve_fit(lin, x[lower_bound:upper_bound], y[lower_bound:upper_bound])
    return params

def Transmission_approx(freq, H_0, r):
    params, cov = curve_fit(Transfer_function_three_slabs, freq, H_0, r)
    return params