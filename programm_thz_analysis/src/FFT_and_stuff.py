from mathTHz import *
import numpy as np
import os
import matplotlib.pyplot as plt

#######################################################################################################

directory = "/20240301" # desired directory to make plots from

#######################################################################################################
#######################################################################################################

directory = os.getcwd() + "/data" + directory #how to get all files names in a directory?
datas = []
i = 0

with os.scandir(directory) as it: # scan directory
    for entry in it:
        if entry.name.endswith(".txt"): # search for every file that ends with .txt
            datas.append([entry.name ,np.genfromtxt(entry.path, comments="#", delimiter="	", skip_header=2)]) # append those files to a big list which includes the data an name of file
            i = i + 1

print("A total of: ",i, " files were red")

fft_datas = [None] * len(datas)
i = 0
for data in datas:
    fft_datas[i] = [data[0], FFT_func(data[1][:,4], data[1][:, 2])] # make FFT of data
    i = i + 1

for fft_data in fft_datas: # plot FFT
    plt.figure()
    plt.xlim(0, 2)
    plt.plot(fft_data[1][0], np.abs(fft_data[1][1]), label='FFT of ' + fft_data[0]) #not sure if the absolut of the FFT here is correct
    plt.ylabel('Intensity / V')
    plt.xlabel(r'$f / THz$')
    plt.legend()
    plt.savefig('build/FFTs/FFT_of_' + fft_data[0] + '.pdf')