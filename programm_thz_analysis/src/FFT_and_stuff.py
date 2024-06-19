from mathTHz import FFT_func
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
#######################################################################################################

directory = "/mnt/c/Users/Max_Koch/Documents/Messungen/CuGeO3/CuGeO3/B_hor" # desired directory to make plots from
measurment_devices = ["Lock-in", "PCA"] #The device that was used for the measurment 
device = measurment_devices[1]
#######################################################################################################
#######################################################################################################

#directory = os.getcwd() + "/CuGeO3" + directory #how to get all files names in a directory?
datas = []
i = 0

with os.scandir(directory) as it: # scan directory
    for entry in it:
        if entry.name.endswith(".txt") and entry.name.endswith("_fft.txt")==False: # search for every file that ends with .txt and arent fft datasets
            datas.append([entry.name ,np.genfromtxt(entry.path, comments="#", delimiter="	", skip_header=2)]) # append those files to a big list which includes the data an name of file
            i = i + 1

print("A total of: ",i, " files were red")

fft_datas = [None] * len(datas)
i = 0
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
        i = i + 1
    if device == measurment_devices[1]:
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
        i = i + 1

for fft_data in fft_datas: # plot FFT and save
    with open('build/FFTs/FFT_of_' + fft_data[0] + '.csv', 'w') as file:
        for line in fft_data[1]:
            file.write(f"{line}\n")
    plt.figure()
    plt.plot(fft_data[1][0], np.abs(fft_data[1][1])/np.abs(np.amax(fft_data[1][1])), label='FFT of ' + fft_data[0]) #not sure if the absolut of the FFT here is correct
    plt.xlim(0, 5)
    plt.yscale("log")
    plt.ylabel('Intensity normalized')
    plt.xlabel(r'$f / THz$')
    plt.legend()
    plt.savefig('build/FFTs/FFT_of_' + fft_data[0] + '.pdf')
    plt.close()