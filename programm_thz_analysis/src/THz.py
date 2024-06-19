import numpy as np
import matplotlib.pyplot as plt 
import csv
from dataclasses import dataclass
from numericsTHz import *
from mathTHz import *
from plot import *
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.signal import savgol_filter
###################################################################################################################################
# Datatypes

@dataclass
class Material_parameters:
    d: float
    n_1: float
    k_1: float 
    n_3: np.array 
    k_3: float

"""
    The algorythm is very unstable depeding on what the starting parameter is.
    For the test data set in which I use "smooth" data this is not the case.
    I Already employ an estimator for the real part of the refractive index, but not for the complex part.
    Even with the estimator the estimation process for other values other than the intial one is quite hard to find at some frequencies.
    Depeding on the epsilon (break condition from step one to step two has to be smaller than epsilon) and the stepsize the apgorythm takes between 15-50 minutes for 1000 frequency values.

    Idea:
        Better preprocessing.
            Usually I give the algo all the data in time domain and filter out too high and too low frequencies afterwards.
            But I think I should try to give the algorythm just a specific timeframe, so that the pre and post pulse will not be inlcuded in the data

    Eliminate Oscillations:
        - Divide by FP
        - Measure over bigger intervall

    """


###################################################################################################################################
###################################################################################################################################
#       All data is read in, in this block.
#       Necessary data are
#       - the time resolved THz measurment
#       - the thickness of the probe
###################################################################################################################################


# The thickness of the probe

d = 1*10**(-3) # thickness of the probe in SI d = 13.5*10**(-6)
n_air = 1.00028
n_slab = 1.00028
k_slab = 0
k_air= 0.000

#n = np.genfromtxt("build/results/LSAT_purged_RT_0.6mm_results.csv", delimiter=", ", comments="#")
#print(n[:,1])

Material = Material_parameters(d = d, n_1=n_air, k_1=k_air, n_3=n_slab, k_3=k_slab)
file_name = "data/07_05_24/ZnTe_purged_better_RT1mm.txt" # filename of the measured data
ref_file_name = 'data/07_05_24/ref_purged.txt' # filename of the reference data
comparison_data_file_name = "data/22_04_24/ZnTe_materialparameters_teramat.txt" # filename of the comaprison data

min_freq = 0.5*10**12
max_freq = 3*10**12

FP = False # Choose if the FabryPerot factor is included in the Transmission function
            # usually for thin samples its better to divide the measured data by the FarbyPerot factor, so in general I never put FP on True
            # As the FP oscillates heavily depending on the probe thickness and that makes the estimation considerably less stable
testing = True
plotting = True # just for plots of measured data, the n k, epsilon and sigma plots will always be made
comparison_parameter = False # if False no comparison data will be red in
filter_0 = False
Ausreisser_raus = False
filter_type = "gaussian"
# HDPE 2070um data/20240409/HDPE_2070um.txt
# 20240409/Si_wafer_rough_700um
# data/22_04_24/ZnTe_1mm.txt
#data/Ca3Co3O8_purged/A_RT_B_RT_Ca3Co3O8.txt
#data/Ca3Co3O8_purged/ref_substrate.txt

if comparison_parameter:    
    parameters = np.genfromtxt(comparison_data_file_name, delimiter="	", comments="#") # The time resolved dataset of the probe measurment
else: 
    parameters = None
#Read the excel file


data_sam = np.genfromtxt(file_name, delimiter="	", comments="#") # The time resolved dataset of the probe measurment
data_ref = np.genfromtxt(ref_file_name,  delimiter="	", comments="#") # the time resolved dataset of the reference measurment

###################################################################################################################################

data_ref[:,0] = data_ref[:,0] * 10**(-12) # ps in seconds
data_sam[:,0] = data_sam[:,0] * 10**(-12)

data_ref[:,0] = data_ref[:,0] + np.abs(np.min(data_ref[:,0])) # move everything to positiv times
data_sam[:,0] = data_sam[:,0] + np.abs(np.min(data_sam[:,0]))

#############################
#maximum1 = find_peaks(data_sam[:,1], height=(0.25))[:2]
#timestep = (maximum1[0][0] - maximum1[0][1])*(data_sam[10,0] - data_sam[11,0])
#d = (c/3.4)*np.abs(2*timestep)
#print(maximum1, d)
#Material.d = d

###################################################################################################################################
#           Filters if wanted  
###################################################################################################################################
if(filter_0):
    x = np.linspace(0,len(data_sam[:,0]),len(data_sam[:,0]))
    plot_gaussian(data_sam[:,0], gaussian(x, find_peaks(data_sam[:,1], prominence=1)[0][0]), data_sam)
    data_ref, data_sam = filter_dataset(data_ref, data_sam, filter=filter_type) #applies filter that was choosen above
else:
    print("No preprocessing filter")

###################################################################################################################################
# This block applies the FFT to the data, aswell as masking frequencies that we dont need for the analization
###################################################################################################################################

freq_ref, amp_ref = FFT_func(data_ref[:,1], data_ref[:,0])  #in Hz
freq_sam, amp_sam = FFT_func(data_sam[:,1], data_sam[:,0])

mask1 = freq_ref < max_freq # mask1ed for THz frequency below 4.5 THz
amp_ref = amp_ref[mask1]
amp_sam = amp_sam[mask1]
freq_ref = freq_ref[mask1]
freq_sam = freq_sam[mask1]
mask2 = min_freq < freq_ref # mask1ed for THz frequency above 200 GHz 200*10**9
amp_ref = amp_ref[mask2]
amp_sam = amp_sam[mask2]
freq_ref = freq_ref[mask2]
freq_sam = freq_sam[mask2]

if comparison_parameter:
    mask_parameter1 = parameters[:,0] < max_freq/10**12
    parameters = parameters[mask_parameter1]
    mask_parameter2 = min_freq/10**12 < parameters[:,0] 
    parameters = parameters[mask_parameter2]

###################################################################################################################################
# This block calculates the complex transfer function and does the unwrapping porcess
###################################################################################################################################

H_0_value = amp_sam/amp_ref # complex transfer function
#H_0_value = lowpass(H_0_value, 0.5)
#H_0_value = savgol_filter(H_0_value.real, len(H_0_value), len(H_0_value)//4) + 1j*savgol_filter(H_0_value.imag, len(H_0_value), len(H_0_value)//4)
angle = np.angle(H_0_value) #angle between complex numbers
phase = (np.unwrap(angle))  #phase  

###################################################################################################################################
# This block makes a linear estimation of the phase
###################################################################################################################################

phase_approx  = (linear_approx(freq_ref, phase)[1] * freq_ref)

###################################################################################################################################
#       All the plotting
###################################################################################################################################


if(plotting):
    print("Plotting...\n")
    plot_ref_sam_phase(freq_ref, amp_ref, amp_sam)
    plot_Intensity_against_time(data_ref, data_sam)
    plot_FFT(freq_ref, amp_ref, freq_ref, amp_sam)
    #plot_phase_against_freq(freq_ref, phase, angle)
    plot_phase_against_freq(freq_ref, phase, angle, zeropadded=False, approx=True, phase_approx=phase_approx)
    plot_realpart_refractive_index(freq_ref, estimater_n(phase, freq_ref, Material))
    plot_complex_refrective_index(freq_ref, estimater_k(freq_ref, H_0_value, estimater_n(phase, freq_ref, Material), Material))
    plot_H_0_against_freq(freq_ref, np.abs(H_0_value))
    plot_Transferfunction(estimater_n(phase, freq_ref, Material), Transfer_function_three_slabs(freq_ref[100], estimater_n(phase, freq_ref, Material), 0.001, Material, FP=False))

###################################################################################################################################
# Here Starts the numerical process of finding the refractive index
###################################################################################################################################

print("-----------------------------------------------------------")

###################################################################################################################################
# Set starting values for the algorythm
###################################################################################################################################
minlimit = 1 # lower limit of the frequency range. 1 for no lower limit.
maxlimit = -2 # upper limit of the frequency range. -2 for no upper limit. 

n_0 = estimater_n(phase, freq_ref, Material, substrate=n_air)[maxlimit] #intial guess for n, calculated as in the paper "A Reliable Method for Extraction of Material
                                                       #Parameters in Terahertz Time-Domain Spectroscopy
                                                       #Lionel Duvillaret, Frédéric Garet, and Jean-Louis Coutaz"
k_0 = estimater_k(freq_ref, H_0_value, estimater_n(phase, freq_ref, Material, substrate=1), Material)[maxlimit] # initial guess for k
r_0 = np.array([n_0,k_0]) # r_p[0] = n, r_p[1] = k 

###################################################################################################################################

r_per_freq = [None]*(len(freq_ref)) # all the n and k per frequency will be written into this array
plt.figure()

plt.plot(freq_ref, Fabry_Perot(freq_ref, [3, 0.03], Material).real, label="real")
plt.plot(freq_ref, Fabry_Perot(freq_ref, [3, 0.03], Material).imag, label="imag")
plt.xlabel("freq")
plt.ylabel("FP")
plt.legend()
plt.savefig("build/testing/Fabryperottests.pdf")
plt.close()

print("starting values for Newton-Raphson: r_per_step =", r_0)
print("SNR T: ", signaltonoise(np.abs(H_0_value)), "dB, SNR arg(T): ", signaltonoise(np.abs(phase)), "dB")
"""
If the material can be considered optically thick we can discard the Fabry Perot factor as reflections inside the material will not travel from on side to the other fast enough.
However, if the material is thin we have to take the FP into account which means that we have to find the right value for FP, n and k.
For this the idea is:
    1. estimate n and k roughly (should already be done with the estimator function)
    2. estimate a FP value 
    3. Divide H_0 (The transmission function) by FP
    4. Do the same process as if the material would be thick and find a n and k
    5. Start the process over until a good value for n and k is found
"""
if(Material.d >= 10**-3 or (filter_type=="truncate" and filter_0 == True)): # If the probe is thick enough (in generall 1mm should be thick enough) we can disregard the FP factor because the time delay between reflections and actual pulse is big enough
    for freq in tqdm((reverse_array(freq_ref[minlimit:maxlimit]))): #walk through frequency range from upper to lower limit
        index = np.argwhere(freq_ref==freq)[0][0] # as we walk though the freq reverse we also have to walk through the other parameters in reverse. So we need to find the corrct index
        params_delta_function = [H_0_value[index], phase_approx[index], freq_ref, index, Material, FP] # we save all the parameters that the error function needs in a big list
        res = minimize(delta_of_r_whole_frequency_range, r_0, bounds=((1, None), (None, None)), args=params_delta_function) # minimizer for the errorfunction. depeding on the method choosen this needs a hess and jac aswell but the basic one is fine without
        if(res.success != True): # if the minimizer cant minize we output an error message
            print("Warning minimizer couldnt terminate: ")
            print(res.message)
            print("At ", freq/10**12, " THz")
            ns = np.linspace(r_0[0]-2, r_0[0]+2, 1000)
            ks = np.linspace(r_0[1], r_0[1], 1000)
            delta = [None]*len(ns)
            i = 0
            for n in ns:
                delta[i] = delta_of_r_whole_frequency_range([n, ks[i]], params_delta_function)
                i = i + 1
            plt.figure()
            plt.plot(ns, delta, label="delta")
            plt.title(str(r_0[0]))
            plt.xlabel("n")
            plt.ylabel("delta")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/delta/deltafunction_for_n_at" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()
            ns = np.linspace(r_0[0], r_0[0], 1000)
            ks = np.linspace(r_0[1]-1, r_0[1]+1, 1000)
            delta = [None]*len(ns)
            i = 0
            for k in ks:
                delta[i] = delta_of_r_whole_frequency_range([ns[i], k], params_delta_function)
                i = i + 1
            plt.figure()
            plt.plot(ks, delta, label="delta")
            plt.title(str(r_0[0]))
            plt.xlabel("k")
            plt.ylabel("delta")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/delta/deltafunction_for_k_at" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()
        r_0 = res.x # otherwise we can just save the result in a temp variable for later use
        if(np.mod(index, 10)==0 and testing): # every 100 steps we look at the estimated Transferfunction, this is just for performance analyzation
            temp_T = np.abs(Transfer_function_three_slabs(freq_ref, r_0[0], r_0[1], Material, FP))
            plt.figure()
            plt.plot(freq_ref, temp_T, label="T")
            plt.plot(freq_ref, (np.abs(H_0_value)), label="actual H_0")
            plt.title(str(r_0))
            plt.xlabel("freq")
            plt.ylabel("T")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/Transferfunction_iteration_" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()
            ns = np.linspace(r_0[0]-2, r_0[0]+2, 1000)
            ks = np.linspace(r_0[1], r_0[1], 1000)
            delta = [None]*len(ns)
            i = 0
            for n in ns:
                delta[i] = delta_rho([n, ks[i]], params_delta_function)
                i = i + 1
            plt.figure()
            plt.plot(ns, delta, label="delta rho")
            plt.title(str(r_0[0]))
            plt.xlabel("k")
            plt.ylabel("delta rho")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/delta/deltarho_for_n_at" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()
            ns = np.linspace(r_0[0]-2, r_0[0]+2, 1000)
            ks = np.linspace(r_0[1], r_0[1], 1000)
            delta = [None]*len(ns)
            i = 0
            for n in ns:
                delta[i] = delta_phi([n, ks[i]], params_delta_function)
                i = i + 1
            plt.figure()
            plt.plot(ns, delta, label="delta phi")
            plt.title(str(r_0[0]))
            plt.xlabel("k")
            plt.ylabel("delta phi")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/delta/deltaphi_for_n_at" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()
        r_per_freq[index] = [r_0[0], r_0[1]] # save the final result of the Newton method for the frequency freq
else: # optically thin sample need to be treated differently 
    for freq in tqdm((reverse_array(freq_ref[minlimit:maxlimit]))): #walk through frequency range from upper to lower limit
        index = np.argwhere(freq_ref==freq)[0][0]    
        params_delta_function = [H_0_value[index], phase_approx[index], freq_ref, index, Material, FP] # we save all the parameters that the error function needs in a big list
        res = minimize(delta_of_r_whole_frequency_range, r_0, bounds=((1, None), (None, None)), args=params_delta_function) # minimizer for the errorfunction. depeding on the method choosen this needs a hess and jac aswell but the basic one is fine without
        if(res.success != True): # if the minimizer cant minize we output an error message
            print("Warning minimizer couldnt terminate: ")
            print(res.message)
            print("At ", freq/10**12, " THz")
        r_0 = res.x # otherwise we can just save the result in a temp variable for later use
        for i in range(20):
            #############################################################################################################################################################
            # Estimation of new H_0 value without the Farby perot value
            #Material_parameters.d = (c/np.abs(r_0[0] - 1j*r_0[1])**2)*np.abs(timestep)
            #print(Material_parameters.d)
            H_0_value_FP_free = H_0_value/Fabry_Perot(freq_ref, r_0, Material) # we divide the measured data by the Farby perot factor to make free
            phase_FP_free = np.unwrap(np.angle(H_0_value_FP_free))
            phase_approx_FP_free  = linear_approx(freq_ref, phase_FP_free)[1] * freq_ref
            ########################################################################################################################################################################
            params_delta_function = [H_0_value_FP_free[index], phase_approx[index], freq_ref, index, Material, FP]
            res = minimize(delta_of_r_whole_frequency_range, r_0, bounds=((1, None), (None, None)), args=params_delta_function) # minimizer needs gradient as a function and hessematrix of the delta function.
            if(index == 50):
                plt.figure()
                plt.plot(freq_ref, np.abs(Fabry_Perot(freq_ref, r_0, Material)), label="FP")
                plt.plot(freq_ref, np.abs(H_0_value_FP_free), label="FP free H_0")
                plt.title(str(r_0))
                plt.xlabel("freq")
                plt.ylabel("H_0/FP")
                plt.legend()
                plt.savefig("build/testing/Transfertest_Thz/FP_freeH_0perstep" + str(freq_ref[index]/10**12)+ str(i) + ".pdf")
                plt.close()
            if(res.success != True):
                print("Warning minimizer couldnt terminate: ")
                print(res.message)
                ns = np.linspace(r_0[0]-2, r_0[0]+2, 1000)
                ks = np.linspace(r_0[1], r_0[1], 1000)
                delta = [None]*len(ns)
                i = 0
                for n in ns:
                    delta[i] = delta_of_r_whole_frequency_range([n, ks[i]], params_delta_function)
                    i = i + 1
                plt.figure()
                plt.plot(ns, delta, label="delta")
                plt.title(str(r_0[0]))
                plt.xlabel("n")
                plt.ylabel("delta")
                plt.legend()
                plt.savefig("build/testing/Transfertest_Thz/delta/deltafunction_for_n_at" + str(freq_ref[index]/10**12) + ".pdf")
                plt.close()
            # hess=Hessematrix_minizer
            r_0 = np.abs(res.x)
        if(np.mod(index, 10)==0 and testing):
            temp_T = np.abs(Transfer_function_three_slabs(freq_ref, r_0[0], r_0[1], Material, FP))
            plt.figure()
            plt.plot(freq_ref,temp_T, label="T")
            plt.plot(freq_ref, np.abs(H_0_value), label="actual H_0")
            plt.title(str(r_0))
            plt.xlabel("freq")
            plt.ylabel("T")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/Transferfunction_iteration_" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()

            plt.figure()
            plt.plot(freq_ref,temp_T, label="T")
            plt.plot(freq_ref, np.abs(H_0_value_FP_free), label="FP free H_0")
            plt.title(str(r_0))
            plt.xlabel("freq")
            plt.ylabel("H_0/FP")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/FP_freeH_0" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()

            plt.figure()
            plt.plot(freq_ref,phase_FP_free, label="phase FP free")
            plt.plot(freq_ref, phase_approx_FP_free, label="approx phase FP free")
            plt.title(str(r_0))
            plt.xlabel("freq")
            plt.ylabel("Phi")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/Phaseat" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()

            plt.figure()
            plt.plot(freq_ref,(Fabry_Perot(freq_ref, r_0, Material)).real, label="FP real")
            plt.plot(freq_ref,(Fabry_Perot(freq_ref, r_0, Material)).imag, label="FP imag")
            plt.title(str(r_0))
            plt.xlabel("freq")
            plt.ylabel("FP")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/FPat" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()

            temp_T = np.abs(Transfer_function_three_slabs(freq_ref, r_0[0], r_0[1], Material, FP))
            plt.figure()
            plt.plot(freq_ref, temp_T, label="T")
            plt.plot(freq_ref, (np.abs(H_0_value)), label="actual H_0")
            plt.title(str(r_0))
            plt.xlabel("freq")
            plt.ylabel("T")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/Transferfunction_iteration_" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()

            ns = np.linspace(r_0[0]-2, r_0[0]+2, 1000)
            ks = np.linspace(r_0[1], r_0[1], 1000)
            delta = [None]*len(ns)
            i = 0
            for n in ns:
                delta[i] = delta_rho([n, ks[i]], params_delta_function)
                i = i + 1
            plt.figure()
            plt.plot(ns, delta, label="delta rho")
            plt.title(str(r_0[0]))
            plt.xlabel("k")
            plt.ylabel("delta rho")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/delta/deltarho_for_n_at" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()

            ns = np.linspace(r_0[0]-2, r_0[0]+2, 1000)
            ks = np.linspace(r_0[1], r_0[1], 1000)
            delta = [None]*len(ns)
            i = 0
            for n in ns:
                delta[i] = delta_phi([n, ks[i]], params_delta_function)
                i = i + 1
            plt.figure()
            plt.plot(ns, delta, label="delta phi")
            plt.title(str(r_0[0]))
            plt.xlabel("k")
            plt.ylabel("delta phi")
            plt.legend()
            plt.savefig("build/testing/Transfertest_Thz/delta/deltaphi_for_n_at" + str(freq_ref[index]/10**12) + ".pdf")
            plt.close()
        #if(r_0[0] > 10):
        #    r_0[0] = 1
        r_per_freq[index] = [r_0[0], r_0[1]] # save the final result of the minimizing method for the frequency freq

alpha = absorption_coef(freq_ref[minlimit:maxlimit], (flatten(r_per_freq[minlimit:maxlimit])[1::2]))
print("Done")

#############################################################################################################################################################################################
# I am not quiute sure how I should treat data that is obviously wrong, so one take is to erase those data points and plot just the relasitic data
#############################################################################################################################################################################################
n = np.array(flatten(r_per_freq[minlimit:maxlimit])[0::2]) # first flatten the convoluted array than take every second entrance starting with the zeroth as those are n
k = np.array(flatten(r_per_freq[minlimit:maxlimit])[1::2]) # first flatten the convoluted array than take every second entrance starting with the first as those are k


print("Plotting...")
plt.figure()
if(Ausreisser_raus):
    mask3 = n < 10
    plt.plot(freq_ref[minlimit:maxlimit][mask3]/1e12, n[mask3], marker=".",ls="", label='n') # we have to flatten the array before it plots 
    plt.plot(freq_ref[minlimit:maxlimit][mask3]/1e12, k[mask3], label='k')
else:
    plt.plot(freq_ref[minlimit:maxlimit]/1e12, n, marker=".",ls="", label='n') # we have to flatten the array before it plots 
    plt.plot(freq_ref[minlimit:maxlimit]/1e12, k, label='k')
if comparison_parameter:
    plt.plot(parameters[:,0], parameters[:,1], label="from tera")
#plt.yscale("log")
plt.xlabel(r'$\omega / THz$')
plt.ylabel('value')
plt.title('parameter: start r ' + str([n_0, k_0]))
plt.legend()
plt.savefig('build/frequncy_against_n_k.pdf')
plt.close()

plot_absorption_coefficient(freq_ref[minlimit:maxlimit], alpha, parameters, False)

###################################################################################################################################
# Optical constants
###################################################################################################################################



epsilon_1 = estimater_epsilon_1(n, k)
epsilon_2 = estimater_epsilon_2(n, k)

sigma_1 = estimater_sigma_1(freq_ref[minlimit:maxlimit], n, k)
sigma_2 = estimater_sigma_2(freq_ref[minlimit:maxlimit], n, k)


filenamecut = reverse_array(file_name).find("/")
file_name = file_name[-filenamecut:-4] # just take the name

output_data = [freq_ref[minlimit:maxlimit], n, k, alpha, epsilon_1, epsilon_2, sigma_1, sigma_2]

with open('build/results/' + file_name + '_results.csv', 'w') as file:
    file.write(str("#freq/Hz,    n,   k,   alpha,   epsilon1,    epsilon2,    sigma1,  sigma2") + "\n")
    for i in range(len(output_data[0])):
        file.write(str([data[i] for data in output_data]) + "\n")

plot_epsilon(freq_ref[minlimit:maxlimit], epsilon_1, epsilon_2)
plot_sigma(freq_ref[minlimit:maxlimit], sigma_1, sigma_2)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.tight_layout()
ax1.plot(freq_ref[minlimit:maxlimit]/1e12, n, label='n')
ax1.plot(freq_ref[minlimit:maxlimit]/1e12, k, label='k')
ax1.set_xlabel(r"$\omega/THz$")
ax1.set_title("n and k")

ax2.plot(freq_ref[minlimit:maxlimit], epsilon_1, label = r"$real\, part\,\epsilon$")
ax2.plot(freq_ref[minlimit:maxlimit], epsilon_2,label=r"$complex\, part\,\epsilon$")
ax2.set_xlabel(r"$\omega/THz$")
ax2.set_title(r"$ \epsilon$")

ax3.plot(freq_ref[minlimit:maxlimit], sigma_1, label = r"$real\, part\,\sigma$")
ax3.plot(freq_ref[minlimit:maxlimit], sigma_2,label=r"$complex\, part\,\sigma$")
ax3.set_xlabel(r"$\omega/THz$")
ax3.set_title(r"$ simga$")

fig.savefig("build/n_k_epsi_sigma.pdf")
plt.close(fig)