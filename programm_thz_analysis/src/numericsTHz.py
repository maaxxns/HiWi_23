import numpy as np
import csv

def grad_2D(func, r, params=None, h = 10**(-6)): 
    grad_0_x = (func([r[0] + h, r[1]], params) - func([r[0] - h, r[1]], params))/2*h
    grad_0_y = (func([r[0],r[1] + h], params) - func([r[0], r[1] - h], params))/2*h 
    return [grad_0_x, grad_0_y]

def Hessematrix(func, r, params=None, h = 10**(-6)):  
    A = (func([r[0] + h, r[1]], params) - 2*func([r[0], r[1]], params) + func([r[0] - h, r[1]], params))/h**2
    B = (func([r[0] + h/2, r[1] + h/2], params) - func([r[0] + h/2, r[1] - h/2], params) - func([r[0] - h/2, r[1] + h/2], params) + func([r[0] - h/2, r[1] - h/2], params))/h**2
    C = B # Satz von Schwarz
    D = (func([r[0], r[1] + h], params) - 2*func([r[0], r[1]], params) + func([r[0], r[1] - h], params))/h**2
    # A = d²delta(r_p)/dn² = (delta(n + h, k) - 2*delta(n,k) - delta(n-h,k))/h²
    # B = d²delta(r_p)/dkdn = (delta(n + h/2, k + h/2) - delta(n + h/2, k - h/2) - delta(n - h/2, k + h/2)  + delta(n - h/2, k - h/2))/h²
    # D = d²delta(r_p)/dk² = (delta(n, k + h) - 2*delta(n,k) - delta(n,k - h))/h²
    return np.array([[A,B], [C,D]])

def newton_r_p_zero_finder(func, r, params = None, h=10**(-6)): #newton iteration step to find zero crossing 
    grad_ = grad_2D(func = func, r = r, params = params, h = h) # calculate the gradient of func(r_p)
    if isinstance(grad_[0], complex) or isinstance(grad_[1], complex):
        print("complex grad")
    with open('build/testing/gradient.csv', 'w') as csvfile: #save the gradient to see if its always negative (we want to get to a minimum or even better to a zero crossing)
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(grad_) 
    print(func(r, params))
  
    r_p_1 = r - func(r, params)/grad_

    return r_p_1 # returns new values for [n_2,k_2] that minimize the error according to newton iteration step 

def newton_minimizer(func, r, params, h=10**(-6)): #newton iteration step to find the best value of r=(n_2,k_2)  
    A = Hessematrix(func, r, params, h) # Calculate the hessian matrix of delta(r_p)
    grad_ = grad_2D(func,r, params, h) # calculate the gradient of delta(r_p)
    X_1 = np.linalg.inv((A))
    X = np.linalg.inv((A)).dot((grad_))
    r_p_1 = r - np.linalg.inv((A)).dot((grad_)) #why is r_p going in negativ direction when both the hesse and the gradient are negativ, should the r_p move in positiv direction than?
    #         ^
    #         I
    #     This + is weird and shouldnt be there but its seems to work with it
    # r_p+1 = r_p - A⁽⁻¹⁾*grad(delta(r_p))
    return r_p_1 # returns new values for [n_2,k_2] that minimize the error according to newton iteration step 
