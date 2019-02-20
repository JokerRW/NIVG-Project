# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:54:30 2018

@author: RICHARD.WENG
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy import linspace
import matplotlib.pyplot as plt
import peakutils 

# -----------setting parameters ---------------

# set source path
path = 'C:\\Users\\richard.weng\\Pictures\\gtc2018\\rawdata_diabetes.txt'
# set save path
path_save = 'C:\\Users\\richard.weng\\Pictures\\gtc2018\\rawdata_diabetes_precessed.txt'
# set save path for peak_valley
path_save_peakValley = 'C:\\Users\\richard.weng\\Pictures\\gtc2018\\rawdata_diabetes_peakValleys.txt'
# set the window size for filtering out low frequencies, the bigger the size, the more low frequencies would be block out
window_size = 0
# set the winodw size2 for filtering out high frequencies
window_size2 = 0
# set the viewing window for display
viewing_window = 8000

# ------------end of setteing ---------------


# read-in txt file
file1 = open(path, 'r',encoding = 'utf-8-sig')
print ("hello glucose")

# create a list for raw data storage
list_diabetes = []

# put data into the raw data list
for line in file1:
    list_diabetes.append(float(line.rstrip('\n')))

# plotting the raw data
plt.plot(list_diabetes[:viewing_window], 'r')
plt.show()
print("-----------------------------------------")
# do fft
ffted_list_diabetes = np.fft.fft(list_diabetes)
print (ffted_list_diabetes)
print("-----------------------------------------")
# set low frequencies to zero
if window_size >0:
    ffted_list_diabetes[:window_size] = 0
    ffted_list_diabetes[-window_size:] = 0
else:
    pass
# set high frequencies to zero
if window_size2 >0:
    ffted_list_diabetes[window_size2:-window_size2] = 0
else:
    pass
print (ffted_list_diabetes)
np.savetxt("ffted_list_diabetes.txt", ffted_list_diabetes, fmt="%s")
print("-----------------------------------------")
# do ifft
cut_ifft_ffted_list_diabetes = np.fft.ifft(ffted_list_diabetes)
np.savetxt("cut_ifft_ffted_list_diabetes.txt", cut_ifft_ffted_list_diabetes, fmt="%s")

# find the peaks
indexes = peakutils.indexes(np.array(cut_ifft_ffted_list_diabetes), thres=0.0001/max(np.array(cut_ifft_ffted_list_diabetes)), min_dist=40)
peak_valley_indexes = []
count = 0
index_pre = []
for ii in indexes:
    if count == 0:
        index_pre = ii
        count = count +1
    else:
        interval = cut_ifft_ffted_list_diabetes.real[index_pre:ii].tolist()
        peak_valley_indexes.append(interval.index(min(interval))+index_pre)
        peak_valley_indexes.append(ii)
        index_pre = ii
        count = count +1

# plot the processed data
time = linspace(1,viewing_window,viewing_window)
indexes = np.asarray(peak_valley_indexes)
plt.plot(time[indexes[indexes<viewing_window]],(cut_ifft_ffted_list_diabetes)[indexes[indexes<viewing_window]],'bo')
plt.plot(time,cut_ifft_ffted_list_diabetes[:viewing_window], 'r')
plt.show()

# save the processed data
fileObject = open(path_save,"w")
for item in cut_ifft_ffted_list_diabetes.real:
    fileObject.write("%s\n" % item)
fileObject.close()

# save the peak valley data
fileObject = open(path_save_peakValley,"w")
for ii in peak_valley_indexes:
    fileObject.write("%s\n" % cut_ifft_ffted_list_diabetes[ii].real)
fileObject.close()
