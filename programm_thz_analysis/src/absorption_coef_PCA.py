import numpy as np
from mathTHz import FFT_func, flatten
from numericsTHz import filter_dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import seaborn as sns

#######################################################################################################

# What should be plotted?
# FFT of all the data in seperated plots
FFT = False
# absorption coef of all measurments in seperated plots
absorb = False
# all absorption coed in one plot
absorb_all_in_one = True

directory = "/mnt/c/Users/Max_Koch/Documents/Messungen/CuGeO3/CuGeO3/B_hor" # desired directory to make plots from
measurment_devices = ["Lock-in", "PCA", "Magnetcryo"] #The device that was used for the measurment 
device = measurment_devices[1]
#######################################################################################################
#######################################################################################################

def alpha(FFT_sam, FFT_ref, d=10**-3): # FFT_data-> (omega, Ampl)
    alpha = -np.log(np.abs(FFT_sam[1])**2/np.abs(FFT_ref)**2) * 1/d /100
    return [FFT_sam[0], alpha] # returns [omega, alpha]

def get_temp_from_filename(name):
    name = name.split("_")
    for s in name:
        test = s.strip("kK")
        if(test.isdigit()):
            temp = test
    return temp

def get_Bfield_from_filename(name): # This needs to be tested
    name = name.split("_")
    for s in name:
        if "T" in s:
            test = s.strip("T")
            if(test.isdigit()):
                temp = test
    return temp

def find_ref_dataset(datas):
    n = 0
    for data in datas:
        if "ref" in data[0] or "Ref" in data[0]:
            n = n + 1
            if n > 1:
                print("more than one reference set found. References sets will be averaged.")
                ref[:, 1] = (ref[:, 1] + data[1])/n
            else:
                print("Using " + data[0] + " as reference dataset")
                ref = data
    return ref

def ps_to_sec(datas):
    for data in datas:
        data[1][:,0] = data[1][:,0]*10**-12
    return datas

#######################################################################################################

#directory = os.getcwd() + "/CuGeO3" + directory #how to get all files names in a directory?
datas = []
i = 0
filter_ = "truncate"

with os.scandir(directory) as it: # scan directory
    for entry in it:
        if entry.name.endswith(".txt") and entry.name.endswith("_fft.txt")==False: # search for every file that ends with .txt and arent fft datasets
            datas.append([entry.name ,np.genfromtxt(entry.path, comments="#", delimiter="	", skip_header=2)]) # append those files to a big list which includes the data an name of file
            i = i + 1

datas = ps_to_sec(datas)
ref_timedomain = find_ref_dataset(datas)

if filter_ == "truncate" or filter_ == "gaussian":
    for data in datas:
        if "ref" in data[0] or "Ref" in data[0]:
            data[1], _ = filter_dataset(ref_timedomain[1], data[1], filter=filter_)
            ref_timedomain = data
        else:
            _, data[1] = filter_dataset(ref_timedomain[1], data[1], filter=filter_)

print("A total of: ",i, " files were red and the filter: " + filter_ + " was applied.")

fft_datas = [None] * len(datas)
i = 0
n = 0
unevenspacing = False
for data in datas:
    if device == measurment_devices[0]:
        # check if data is actually evenly spaced otherwise we have to do a interpolation
        t = np.abs(data[1][1, 4] - data[1][0, 4]) 
        for i in np.arange(len(data[1][:, 4]) - 1):
            spacing = data[1][i + 1, 4] - data[1][i, 4]
            if(np.abs(spacing - t) > 10**-6):
                unevenspacing = True
                print("Datapoints seem to be unevenly spaced, interpolation will be initiated")
        if(unevenspacing):
            spl = CubicSpline(data[1][:,4], data[1][:, 2])
            xnew = np.linspace(data[1][0, 4], data[1][-1, 4], len(data[1][:,4])*5) # lets use five times the amount of "old" datapoints for the spline
            fft_datas[i] = [data[0], FFT_func(spl(xnew), xnew)] 
        else:
            fft_datas[i] = [data[0], FFT_func(data[1][:,4], data[1][:, 2])] # make FFT of data
    if device == measurment_devices[1] or device == measurment_devices[2]:
                # check if data is actually evenly spaced otherwise we have to do a interpolation
        t = np.abs(data[1][1, 1] - data[1][0, 1]) 
        #for i in np.arange(len(data[1][:, 1]) - 1):
        #    spacing = data[1][i + 1, 1] - data[1][i, 1]
        #    if(np.abs(spacing - t) > 10**-5):
        #        unevenspacing = True
        #        print("Datapoints seem to be unevenly spaced, interpolation will be initiated")
        if(unevenspacing):
            spl = CubicSpline(data[1][:,0], data[1][:, 1])
            xnew = np.linspace(data[1][0, 1], data[1][-1, 1], len(data[1][:,1])*5) # lets use five times the amount of "old" datapoints for the spline
            fft_datas[i] = [data[0], FFT_func(spl(xnew), xnew)]
        else:
            fft_datas[i] = [data[0], FFT_func(data[1][:,1], data[1][:, 0])] # make FFT of data
    if "ref" in data[0] or "Ref" in data[0]:
        n = n + 1
        if n > 1:
            print("more than one reference set found. References sets will be averaged.")
            FFT_ref[:, 1] = (FFT_ref[:, 1] + fft_datas[i][1][1])/n
        else :
            print("Using " + fft_datas[i][0] + " as reference dataset")
            FFT_ref = fft_datas[i][1][1]
    i = i + 1

if FFT:
    for fft_data in tqdm(fft_datas): # plot FFT and save
        with open('build/FFTs/FFT_of_' + fft_data[0] + '.csv', 'w') as file:
            for line in fft_data[1]:
                file.write(f"{line}\n")
        plt.figure()
        plt.plot(fft_data[1][0]*10**-12, np.abs(fft_data[1][1])/np.abs(np.amax(fft_data[1][1])), label='FFT of ' + fft_data[0]) #not sure if the absolut of the FFT here is correct
        plt.xlim(0, 5)
        plt.yscale("log")
        plt.ylabel('Amplitude normalized')
        plt.xlabel(r'$\omega / THz$')
        plt.legend()
        plt.savefig('build/FFTs/FFT_of_' + fft_data[0] + '.pdf')
        plt.close()

#################################################################################################################################################

#       Estimation of absorption coef

#################################################################################################################################################

alpha_data = [None] * len(datas)
i = 0
for fft_data in tqdm(fft_datas): # calculate the absorption coef
    if "ref" not in fft_data[0] or "Ref" not in fft_data[0]:
        if device == measurment_devices[1]:
            temperature = get_temp_from_filename(fft_data[0])
            alpha_data[i] = [int(temperature), alpha(fft_data[1], FFT_ref)]#
        if device == measurment_devices[2]:
            B_field = get_Bfield_from_filename(fft_data[0])
            alpha_data[i] = [int(B_field), alpha(fft_data[1], FFT_ref)]#
        if 25 == alpha_data[i][0]:
            alpha_reference = alpha_data[i][1][1]
            print("Using " + str(alpha_data[i][0]) + "K as reference for alpha")
    i = i + 1

if absorb:
        plt.figure()
        if device == measurment_devices[1]:
            plt.plot(fft_data[1][0]*10**-12, alpha_data[i][1][1], label=str(temperature)) #not sure if the absolut of the FFT here is correct
        if device == measurment_devices[2]:
            plt.plot(fft_data[1][0]*10**-12, alpha_data[i][1][1], label=str(B_field)) #not sure if the absolut of the FFT here is correct
        plt.xlim(0, 5)
        plt.ylabel(r'$\alpha / cm$')
        plt.xlabel(r'$\omega / THz$')
        plt.legend()
        plt.savefig('build/FFTs/alpha_' + fft_data[0] + '.pdf')
        plt.close()

fig = plt.figure()
colors = sns.color_palette('icefire', len(alpha_data))
ax = fig.add_subplot(111)
ax.set_prop_cycle('color', colors)
i = 0
lol = flatten(alpha_data)[0::2]
alpha_data = sorted(alpha_data, key=lambda temp: temp[0])
if absorb_all_in_one:
    for alpha_ in tqdm(alpha_data): # calculate the absorption coef
        ax.plot(fft_data[1][0]*10**-12, alpha_[1][1] - alpha_reference + alpha_[0]*3, label=str(alpha_[0])) #not sure if the absolut of the FFT here is correct
        i = i + 1
ax.set_xlim(0.5, 2)
ax.set_ylabel(r'$\alpha / cm$')
ax.set_xlabel(r'$\omega / THz$')
ax.legend(prop={"size": 3})
plt.savefig('build/FFTs/all_alphas.pdf')
plt.close()