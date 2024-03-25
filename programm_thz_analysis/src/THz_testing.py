import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *

@dataclass
class Material_parameters:
    d: float
    n_1: float
    k_1: float 
    n_3: float 
    k_3: float

# The thickness of the probe

d = 0.26*10**(-3) # thickness of the probe in SI
n_air = 1
n_slab = 1
k_slab = 1




Material = Material_parameters(d = d, n_1=n_air, k_1=n_air, n_3=n_slab, k_3=k_slab)

freq = np.linspace(1*10**10, 3*10**12, 1000) #test freq from 10 Ghz to 3 THz


T = Transfer_function_three_slabs(freq, 1 , 2, 1, 1, 2, 1, d, False)

r_p = [1.5,1.5]

steps = np.linspace(0, 100)
params = [T[100], freq[100], Material]

for step in steps:
    r_p = newton_r_p_zero_finder(delta_of_r, r_p, parameter=params, h = 10**-4)
    print(r_p)