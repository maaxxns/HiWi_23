import numpy as np
import matplotlib.pyplot as plt

def plot_Intensity_against_time(data_ref, data_sam): #data_ref is refernece data, data_sam is sample data
    # The function plots the Intensity of the sample and reference in time

    plt.figure()
    plt.plot(data_ref[:,0]*10**(12), data_ref[:,1], label='Reference')
    plt.plot(data_sam[:,0]*10**(12) + 100, data_sam[:,1], label='Sample moved by +100ps')
    plt.xlabel(r'$ t/ps $')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.title('The reference and sample data set')
    plt.savefig('build/THz_timedomain.pdf')
    plt.close()

def plot_Intensity_against_time_zeropadding(data_ref_zero, data_sam_zero):
    #data_ref_zero is refernece data zero padded, data_sam_zero is sample data zeropadded
    # The function plots the Intensity of the sample and reference in time

    plt.figure()
    plt.plot(data_ref_zero[0]*10**(12), data_ref_zero[1], label='Reference')
    plt.plot(data_sam_zero[0]*10**(12) + 100, data_sam_zero[1], label='Sample moved by +100ps')
    plt.xlabel(r'$ t/ps $')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.title('The reference and sample data set with zero padding')
    plt.savefig('build/THz_timedomain_zero.pdf')
    plt.close()

def plot_FFT(freq_ref, amp_ref, freq_sam, amp_sam, zeropadded=False):
    #plots the FFT data of a given spectral amplitude and frequency. If zeropadded is true zeropadded data needs to be given into the function to plots those.
    if(zeropadded):
        plt.figure()
        plt.plot(freq_ref* 10**(-12), np.abs(amp_ref), label='Reference FFT') # plot in Thz
        plt.plot(freq_sam* 10**(-12), np.abs(amp_sam), label='Sample FFT')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel('Spectral Amplitude')
        plt.legend()
        plt.grid()
        plt.title('The FFT of the zero padded data sets')
        plt.savefig('build/THz_FFT_zero.pdf')
        plt.close()
    else:
        plt.figure()
        plt.plot(freq_ref* 10**(-12), np.abs(amp_ref), label='Reference FFT') # plot in Thz
        plt.plot(freq_sam* 10**(-12), np.abs(amp_sam), label='Sample FFT')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel('Spectral Amplitude')
        plt.legend()
        plt.grid()
        plt.title('The FFT of the data set')
        plt.savefig('build/THz_FFT.pdf')
        plt.close()

def plot_phase_against_freq(freq_ref, phase, angle, zeropadded=False, approx=False, phase_approx=None,):
    #plots phase and angle of a complex function against time. The phase should be te unwrapped angle value.
    if(zeropadded):
        plt.figure()
        #plt.plot(freq_ref_zero*10**(-12),angle_zero, label='angle')
        plt.plot(freq_ref*10**(-12),phase, label='phase directly from H_0')
        #plt.plot(data_ref[:,0], filter_ref)
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$\Phi/rad$')
        plt.legend()
        plt.grid()
        plt.title('The phase unwarpped of the zero padded data')
        plt.savefig('build/THzPhase_zero.pdf')
        plt.close()
    elif(approx):
        plt.figure()
        plt.plot(freq_ref*10**(-12),phase, label='phase unwrapped')
        plt.plot(freq_ref*10**(-12),angle, label='angle directly from H_0')
        plt.plot(freq_ref*10**(-12),phase_approx, label='phase approximation')
        #plt.plot(data_ref[:,0], filter_ref)
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$\Phi/rad$')

        plt.legend()
        plt.grid()
        plt.title('The phase unwarpped and wrapped and approximated')
        plt.savefig('build/THzPhase_approximation.pdf')
        plt.close()
    else:
        plt.figure()
        plt.plot(freq_ref*10**(-12),phase, label='phase unwrapped')
        plt.plot(freq_ref*10**(-12),angle, label='angle directly from H_0')
        #plt.plot(data_ref[:,0], filter_ref)
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$\Phi/rad$')
        plt.legend()
        plt.grid()
        plt.title('The phase unwarpped and wrapped')
        plt.savefig('build/THzPhase.pdf')
        plt.close()
    
def plot_realpart_refractive_index(freq_ref, n_real, zeropadded=False):    
    if(zeropadded):
        plt.figure()
        #plt.plot(freq_ref*10**(-12), n(freq_ref, d, phase_dif), label='real part of refractive index calculated by the alternative unwrap')
        plt.plot(freq_ref*10**(-12), n_real, label='real part of refractive index')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel('n (arb.)')
        plt.legend()
        plt.grid()
        plt.title('The real part of the refractive index from zero padded data')
        plt.savefig('build/THz_real_index_zero.pdf')
        plt.close()
    else:
        plt.figure()
        #plt.plot(freq_ref*10**(-12), n(freq_ref, d, phase_dif), label='real part of refractive index calculated by the alternative unwrap')
        plt.plot(freq_ref*10**(-12), n_real, label='real part of refractive index')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel('n (arb.)')
        plt.legend()
        plt.grid()
        plt.title('The real part of the refractive index')
        plt.savefig('build/THz_real_index.pdf')
        plt.close()

def plot_complex_refrective_index(freq_ref, n_im, zeropadded=False):
    if(zeropadded):
        plt.figure()
        #plt.plot(freq_ref*10**(-12), k(freq_ref, d, H_0_value, n_real_alt), label='complex part of refractive index by alt')
        plt.plot(freq_ref*10**(-12), n_im, label='complex part of refractive index')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel('k (arb.)')
        plt.legend()
        plt.grid()
        plt.title('The complex part of the refractive index from zero padded data')
        plt.savefig('build/THz_complex_index_zero.pdf')
        plt.close()
    else:
        plt.figure()
        #plt.plot(freq_ref*10**(-12), k(freq_ref, d, H_0_value, n_real_alt), label='complex part of refractive index by alt')
        plt.plot(freq_ref*10**(-12), n_im, label='complex part of refractive index')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel('k (arb.)')
        plt.legend()
        plt.grid()
        plt.title('The complex part of the refractive index')
        plt.savefig('build/THz_complex_index.pdf')
        plt.close()

def plot_absorption_coefficient(freq_ref, alpha, parameters=None, zeropadded=False):
    if(zeropadded):
        plt.figure()
        plt.plot(freq_ref*10**(-12), alpha, label='Absorption coefficient')
        if(parameters != None):
            plt.plot(parameters[:,0], parameters[:,2], label="absorptioncoeffecient from tera")
        #plt.plot(data_ref[:,0], filter_ref)
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$\alpha/cm$')
        plt.legend()
        plt.grid()
        plt.title('The absorption coeffiecient of silicon for zero padded data')
        plt.savefig('build/THz_absorption_zero.pdf')
        plt.close()
    else:
        plt.figure()
        plt.plot(freq_ref*10**(-12), alpha, label='Absorption coefficient')
        if(parameters is not None):
            plt.plot(parameters[:,0], parameters[:,2], label="absorptioncoeffecient from tera")
        #plt.plot(data_ref[:,0], filter_ref)
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$\alpha/cm$')
        plt.legend()
        plt.grid()
        plt.title('The absorption coeffiecient')
        plt.savefig('build/THz_absorption.pdf')
        plt.close()

def plot_gaussian(x,y, data):
        plt.figure()
        plt.plot(x, y, label='gaussian')
        plt.plot(data[:,0], data[:,1], label="data")
        plt.xlabel(r'$ t/ps $')
        plt.ylabel(r'$I$')
        plt.legend()
        plt.grid()
        plt.title('gaussian')
        plt.savefig('build/gaussian.pdf')
        plt.close()

def plot_H_0_against_freq(freq, H_0, zeropadded=False):
    if(zeropadded):
        plt.figure()
        plt.plot(freq/10**12, H_0, label='H_0')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$T$')
        plt.legend()
        plt.grid()
        plt.title('Transmissionfunction zeropadded in freq')
        plt.savefig('build/Transmissionfunction_zeropadded.pdf')
        plt.close()
    else:
        plt.figure()
        plt.plot(freq/10**12, H_0, label='H_0')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$T$')
        plt.legend()
        plt.grid()
        plt.title('Transmissionfunction in freq')
        plt.savefig('build/Transmissionfunction.pdf')
        plt.close()

def plot_FabryPerot(freq, fabry):
    plt.figure()
    plt.plot(freq/10**12, fabry.real, label='real')
    plt.plot(freq/10**12, fabry.imag, label='imag')
    plt.xlabel(r'$ \omega/THz $')
    plt.ylabel(r'$Fabry Perot$')
    plt.legend()
    plt.grid()
    plt.title('FabryPerot in freq')
    plt.savefig('build/FabryPerot.pdf')
    plt.close()

def plot_epsilon(freq, epsilon_1, epsilon_2):
        plt.figure()
        plt.plot(freq/10**12, epsilon_1, label=r'$real\, part\, \epsilon$')
        plt.plot(freq/10**12, epsilon_2, label=r'$complex\, part\, \epsilon$')
        plt.xlabel(r'$ \omega/THz $')
        plt.ylabel(r'$\epsilon$')
        plt.legend()
        plt.grid()
        plt.title('Dieelectric function against freq')
        plt.savefig('build/epsilon.pdf')
        plt.close()

def plot_sigma(freq, sigma1, sigma2):
    plt.figure()
    plt.plot(freq/10**12, np.abs(sigma1), label=r'$real\, part\, \sigma$')
    plt.plot(freq/10**12, np.abs(sigma2), label=r'$complex\, part\, \sigma$')
    plt.yscale('log')
    plt.xlabel(r'$ \omega/THz $')
    plt.ylabel(r'$\sigma$')
    plt.legend()
    plt.grid()
    plt.title('conductivity against freq')
    plt.savefig('build/sigma.pdf')
    plt.close() 

def plot_ref_sam_phase(freq_ref, amp_ref, amp_sam):
    plt.figure()
    plt.plot(freq_ref/10**12, np.unwrap(np.angle(amp_ref)), label="phase of reference")
    plt.plot(freq_ref/10**12, np.unwrap(np.angle(amp_sam)), label="phase of sample")
    plt.legend()
    plt.title("The reference and sample phase against frequency")
    plt.xlabel(r"$\omega/THz$")
    plt.ylabel(r"$Phase/rad$")
    plt.savefig("build/ref_sam_phase.pdf")
    plt.close()

def plot_phase_against_freq_debug(freq_ref, phase, angle, index, i):
    #plots phase and angle of a complex function against time. The phase should be te unwrapped angle value.
    plt.figure()
    #plt.plot(freq_ref_zero*10**(-12),angle_zero, label='angle')
    plt.plot(freq_ref*10**(-12),angle, label='angle directly from H_0')
    plt.plot(freq_ref*10**(-12),phase, label='phase directly from H_0')
    #plt.plot(data_ref[:,0], filter_ref)
    plt.xlabel(r'$ \omega/THz $')
    plt.ylabel(r'$\Phi/rad$')
    plt.legend()
    plt.grid()
    plt.title('The phase unwarpped of the zero padded data')
    plt.savefig('build/testing/Transfertest_Thz/Phase/THzPhase' + str(index) +','+ str(i) +'.pdf')
    plt.close()

def plot_Transferfunction(n, T):
    plt.figure()
    plt.plot(n, T.real, label=r'$real\, part\, T$')
    plt.plot(n, T.imag, label=r'$complex\, part\, T$')
    plt.xlabel(r'$ \omega/THz $')
    plt.ylabel(r'$T$')
    plt.legend()
    plt.grid()
    plt.title('T against n')
    plt.savefig('build/Transferfunctionagainstnm.pdf')
    plt.close() 

#def plot_Transferfunction_against_n(r, T, params):
#    n = r[0]
#    k = r[1]
#    H_0_measured = params[0]
#    phase_mes = params[1]
#    freq = params[2]
#    index = params[3]
#    Material_parameter = params[4]
#    FP = params[5]
#    ns = np.linspace(r[0]-2, r[0]+2, 300)
#    ks = np.linspace(r[1], r[1], 300)
#    plt.figure()
#    plt.plot(ns, np.abs(Transfer_function_three_slabs(freq, ns, ks[1], Material_parameter, FP=None)), label="Phase approx")
#    plt.title(str(r[0]))
#    plt.xlabel("n")
#    plt.ylabel("transferfunction absolut")
#    plt.legend()
#    plt.savefig("build/testing/Transfertest_Thz/delta/Transferfunction_against_n" + str(freq[index]/10**12) + ".pdf")
#    plt.close()