# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:09:28 2018

@author: RICHARD.WENG
"""

import os
import numpy as np
from numpy.fft import fftfreq, fftshift
from scipy import linspace
import matplotlib.pyplot as plt
import peakutils
import pandas as pd
from pandas import DataFrame
from pandas import ExcelWriter
import tkinter as tk

def all_files_transfer():
    #讀取所有xls file的檔案
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    xls_file = []
    for f in files:
        if f[-3:] == 'xls':
            xls_file.append(f)   
    print('待處理的檔案:',xls_file)

    #將所有xls file的檔案更名為txt檔
    for f in xls_file:
        base = os.path.splitext(f)[0]
        os.rename(f, base + ".txt")

def file_FFT(f):
    doc = open(f, 'r')
    lines = doc.readlines()
    newlines = []
    for line in lines:
        # 確認只要6 column 的數才要
        if len(line.split('\t')) == 6:        
        # 將 line 加到新的 newlines 中
            newlines.append(line.split())
    df = pd.DataFrame(newlines,columns = ['Ch : LED 2','Ch : LED 3','Ch : LED 1','x','xx','xxx'] )
    #只要前3 column
    df = df.iloc[:,0:3]
    #调整列顺序
    df = df[['Ch : LED 1','Ch : LED 2','Ch : LED 3']]
    #Create list for AC and DC processing
    led1_list = df['Ch : LED 1'].tolist()
    led2_list = df['Ch : LED 2'].tolist()
    led3_list = df['Ch : LED 3'].tolist()
    led1_list = [float(i) for i in led1_list]
    led2_list = [float(i) for i in led2_list]
    led3_list = [float(i) for i in led3_list]

    # -----------setting parameters for AC data collection---------------

    # set the window size for filtering out low frequencies, the bigger the size, the more low frequencies would be block out
    # N次/80second = xHz, 所以N次= 80second*xHz
    freqHz = 1
    window_size = int(80*freqHz)
    # set the winodw size for filtering out high frequencies
    freqHz1 = 5
    window_size1 = int(80*freqHz1)

    # -----------setting parameters for DC data collection---------------

    # set the window size for filtering out low frequencies, the bigger the size, the more low frequencies would be block out
    freqHz2 = 0.01
    window_size2 = int(80*freqHz2)
    # set the winodw size for filtering out high frequencies
    freqHz3 = 0.1
    window_size3 = int(80*freqHz3)
    viewing_window = 8000

    # ------------end of setteing ---------------

    # plotting the raw data
    plt.figure(figsize=(20,3))
    plt.subplot(131)
    plt.plot(led1_list[:viewing_window], 'r')
    plt.title('940nm Original Voltage response (light transmittance)')
    plt.xlabel('time series (0.01s/dot)')
    plt.ylabel('Voltage')
    plt.subplot(132)
    plt.plot(led2_list[:viewing_window], 'r')
    plt.title('970nm Original Voltage response (light transmittance)')
    plt.xlabel('time series (0.01s/dot)')
    plt.ylabel('Voltage')
    plt.subplot(133)
    plt.plot(led3_list[:viewing_window], 'r')
    plt.title('1200nm Original Voltage response (light transmittance)')
    plt.xlabel('time series (0.01s/dot)')
    plt.ylabel('Voltage')
    plt.show()
    print("-------------------------------------------------------------------------------------------------")

    # do fft (Create name of AC and DC for processing)
    led1_fft_AC = np.fft.fft(led1_list)
    led1_fft_DC = np.fft.fft(led1_list)
    led2_fft_AC = np.fft.fft(led2_list)
    led2_fft_DC = np.fft.fft(led2_list)
    led3_fft_AC = np.fft.fft(led3_list)
    led3_fft_DC = np.fft.fft(led3_list)

    # -----------filtering Processing for AC data collection---------------

    # set low frequencies to zero
    if window_size >=0:
        led1_fft_AC[:window_size] = 0
        led1_fft_AC[-window_size:] = 0
        led2_fft_AC[:window_size] = 0
        led2_fft_AC[-window_size:] = 0
        led3_fft_AC[:window_size] = 0
        led3_fft_AC[-window_size:] = 0
    else:
        pass

    # set high frequencies to zero
    if window_size1 >=0:
        led1_fft_AC[window_size1:-window_size1] = 0
        led2_fft_AC[window_size1:-window_size1] = 0
        led3_fft_AC[window_size1:-window_size1] = 0
    else:
        pass

    # -----------filtering Processing for DC data collection---------------

    # set low frequencies to zero
    if window_size2 >0:
        led1_fft_DC[:window_size2] = 0
        led1_fft_DC[-window_size2:] = 0
        led2_fft_DC[:window_size2] = 0
        led2_fft_DC[-window_size2:] = 0
        led3_fft_DC[:window_size2] = 0
        led3_fft_DC[-window_size2:] = 0
    else:
        pass

    # set high frequencies to zero
    if window_size3 >0:
        led1_fft_DC[window_size3:-window_size3] = 0
        led2_fft_DC[window_size3:-window_size3] = 0
        led3_fft_DC[window_size3:-window_size3] = 0
    else:
        pass

    # plot the fft data
    # number of signal points
    N = len(led1_fft_AC)
    # sample spacing
    T = 0.01
    x = np.linspace(0.0, N*T, N)
    yf1 = led1_fft_AC
    yf2 = led2_fft_AC
    yf3 = led3_fft_AC
    yf4 = led1_fft_DC
    yf5 = led2_fft_DC
    yf6 = led3_fft_DC
    ## Get Power Spectral Density
    signalPSD1 = np.abs(fftshift(yf1))
    PSDplot1 = 1.0/N*signalPSD1
    signalPSD2 = np.abs(fftshift(yf2))
    PSDplot2 = 1.0/N*signalPSD2
    signalPSD3 = np.abs(fftshift(yf3))
    PSDplot3 = 1.0/N*signalPSD3
    signalPSD4 = np.abs(fftshift(yf4))
    PSDplot4 = 1.0/N*signalPSD4
    signalPSD5 = np.abs(fftshift(yf5))
    PSDplot5 = 1.0/N*signalPSD5
    signalPSD6 = np.abs(fftshift(yf6))
    PSDplot6 = 1.0/N*signalPSD6

    ## Get frequencies corresponding to signal PSD
    xf = fftfreq(N, T)
    xf = fftshift(xf)

    # Start plot
    plt.figure(figsize=(20,4))
    plt.subplot(231)
    plt.plot(xf, PSDplot1)
    plt.xlim(-0.5,50) #setting x axis range
    plt.ylim(0,0.002) #setting y axis range
    plt.grid()
    plt.title('940 nm AC Frequence vs strength Plot')
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('PSD')
    plt.subplot(232)
    plt.plot(xf, PSDplot2)
    plt.xlim(-0.5,50) #setting x axis range
    plt.ylim(0,0.002) #setting y axis range
    plt.grid()
    plt.title('970 nm AC Frequence vs strength Plot')
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('PSD')
    plt.subplot(233)
    plt.plot(xf, PSDplot3)
    plt.xlim(-0.5,50) #setting x axis range
    plt.ylim(0,0.002) #setting y axis range
    plt.grid()
    plt.title('1200 nm AC Frequence vs strength Plot')
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('PSD')
    plt.subplot(234)
    plt.plot(xf, PSDplot4)
    plt.xlim(-0.5,50) #setting x axis range
    plt.ylim(0,0.002) #setting y axis range
    plt.grid()
    plt.title('940 nm DC Frequence vs strength Plot')
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('PSD')
    plt.subplot(235)
    plt.plot(xf, PSDplot5)
    plt.xlim(-0.5,50) #setting x axis range
    plt.ylim(0,0.002) #setting y axis range
    plt.grid()
    plt.title('970 nm DC Frequence vs strength Plot')
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('PSD')
    plt.subplot(236)
    plt.plot(xf, PSDplot6)
    plt.xlim(-0.5,50) #setting x axis range
    plt.ylim(0,0.002) #setting y axis range
    plt.grid()
    plt.title('1200 nm DC Frequence vs strength Plot')
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('PSD')
    plt.show()
    print("-------------------------------------------------------------------------------------------------")

    # do ifft
    led1_AC_ifft = np.fft.ifft(led1_fft_AC)
    led1_DC_ifft = np.fft.ifft(led1_fft_DC)
    led2_AC_ifft = np.fft.ifft(led2_fft_AC)
    led2_DC_ifft = np.fft.ifft(led2_fft_DC)
    led3_AC_ifft = np.fft.ifft(led3_fft_AC)
    led3_DC_ifft = np.fft.ifft(led3_fft_DC)

    # find the peaks of 3 LED_AC
    # 940 nm find peaks
    indexes_940 = peakutils.indexes(np.array(led1_AC_ifft), thres=0.0001/max(np.array(led1_AC_ifft)), min_dist=40)
    peak_valley_indexes_940 = []
    count_940 = 0
    index_pre_940 = []
    for ii in indexes_940:
        if count_940 == 0:
            index_pre_940 = ii
            count_940 = count_940 +1
        else:
            interval = led1_AC_ifft.real[index_pre_940:ii].tolist()
            peak_valley_indexes_940.append(interval.index(min(interval))+index_pre_940)
            peak_valley_indexes_940.append(ii)
            index_pre_940 = ii
            count_940 = count_940 +1

    # 970 nm find peaks
    indexes_970 = peakutils.indexes(np.array(led2_AC_ifft), thres=0.0001/max(np.array(led2_AC_ifft)), min_dist=40)
    peak_valley_indexes_970 = []
    count_970 = 0
    index_pre_970 = []
    for ii in indexes_970:
        if count_970 == 0:
            index_pre_970 = ii
            count_970 = count_970 +1
        else:
            interval = led2_AC_ifft.real[index_pre_970:ii].tolist()
            peak_valley_indexes_970.append(interval.index(min(interval))+index_pre_970)
            peak_valley_indexes_970.append(ii)
            index_pre_970 = ii
            count_970 = count_970 +1

    # 1200 nm find peaks
    indexes_1200 = peakutils.indexes(np.array(led3_AC_ifft), thres=0.0001/max(np.array(led3_AC_ifft)), min_dist=40)
    peak_valley_indexes_1200 = []
    count_1200 = 0
    index_pre_1200 = []
    for ii in indexes_1200:
        if count_1200 == 0:
            index_pre_1200 = ii
            count_1200 = count_1200 +1
        else:
            interval = led3_AC_ifft.real[index_pre_1200:ii].tolist()
            peak_valley_indexes_1200.append(interval.index(min(interval))+index_pre_1200)
            peak_valley_indexes_1200.append(ii)
            index_pre_1200 = ii
            count_1200 = count_1200 +1

    # plot the processed data
    time = linspace(1,viewing_window,viewing_window)
    indexes1 = np.asarray(peak_valley_indexes_940)
    indexes2 = np.asarray(peak_valley_indexes_970)
    indexes3 = np.asarray(peak_valley_indexes_1200)
    plt.figure(figsize=(20,4))
    plt.subplot(131)
    plt.plot(time[indexes1[indexes1<viewing_window]],(led1_AC_ifft)[indexes1[indexes1<viewing_window]],'bo')
    plt.plot(time,led1_AC_ifft[:viewing_window], 'r')
    plt.title('940 nm After fft Voltage response (light transmittance)')
    plt.xlabel('time series (0.01s/dot)')
    plt.ylabel('Voltage')
    plt.ylim(-0.03,0.03) #setting y axis range
    plt.subplot(132)
    plt.plot(time[indexes2[indexes2<viewing_window]],(led2_AC_ifft)[indexes2[indexes2<viewing_window]],'bo')
    plt.plot(time,led2_AC_ifft[:viewing_window], 'r')
    plt.title('970 nm After fft Voltage response (light transmittance)')
    plt.xlabel('time series (0.01s/dot)')
    plt.ylabel('Voltage')
    plt.ylim(-0.03,0.03) #setting y axis range
    plt.subplot(133)
    plt.plot(time[indexes3[indexes3<viewing_window]],(led3_AC_ifft)[indexes3[indexes3<viewing_window]],'bo')
    plt.plot(time,led3_AC_ifft[:viewing_window], 'r')
    plt.title('1200 nm After fft Voltage response (light transmittance)')
    plt.xlabel('time series (0.01s/dot)')
    plt.ylabel('Voltage')
    plt.ylim(-0.03,0.03) #setting y axis range
    plt.show()

    # 重新製作dataframe,將處理好的AC and DC資料放入
    df['940 AC'] = led1_AC_ifft.real
    df['970 AC'] = led2_AC_ifft.real
    df['1200 AC'] = led3_AC_ifft.real
    df['940 DC'] = led1_DC_ifft.real
    df['970 DC'] = led2_DC_ifft.real
    df['1200 DC'] = led3_DC_ifft.real
    # 
    peak_940nm = []
    for ii in peak_valley_indexes_940:
        peak_940nm.append(led1_AC_ifft[ii].real)
    peak_970nm = []
    for ii in peak_valley_indexes_970:
        peak_970nm.append(led2_AC_ifft[ii].real)
    peak_1200nm = []
    for ii in peak_valley_indexes_1200:
        peak_1200nm.append(led3_AC_ifft[ii].real)

    df_peak_940 = DataFrame(peak_940nm, columns = ["940 Peak"])
    df_peak_970 = DataFrame(peak_970nm, columns = ["970 Peak"])
    df_peak_1200 = DataFrame(peak_1200nm, columns = ["1200 Peak"])
    # peak_dic = {"Peak 940nm":peak_940nm,"Peak 970nm":peak_970nm,"Peak 1200nm":peak_1200nm}
    # df2 = DataFrame(peak_dic) (無法使用因為有時候peak數目不同)

    #AC 數據處理, 前三筆不要，取60組的振幅平均
    #移除前三筆數據比
    peak_940nm_3_remove = peak_940nm[3:]
    peak_970nm_3_remove = peak_970nm[3:]
    peak_1200nm_3_remove = peak_1200nm[3:]
    ac_940nm = []
    ac_970nm = []
    ac_1200nm = []
    for i in range(1, 121):
        if i%2 ==1:
            ac_940nm.append(peak_940nm_3_remove[i-1]-peak_940nm_3_remove[i])
            ac_970nm.append(peak_970nm_3_remove[i-1]-peak_970nm_3_remove[i])
            ac_1200nm.append(peak_1200nm_3_remove[i-1]-peak_1200nm_3_remove[i])
    ac_940_result = np.mean(np.array(ac_940nm))
    ac_970_result = np.mean(np.array(ac_970nm))
    ac_1200_result = np.mean(np.array(ac_1200nm))
    #計算心跳
    HeatRate_940 = (len(peak_940nm)//2)/80*60
    HeatRate_970 = (len(peak_970nm)//2)/80*60
    HeatRate_1200 = (len(peak_1200nm)//2)/80*60

    #DC 數據處理，取6000筆數據的平均
    dc_940nm = df.iloc[1001:6001, 6].mean()
    dc_970nm = df.iloc[1001:6001, 7].mean()
    dc_1200nm = df.iloc[1001:6001, 8].mean()

    #Summary df_summary
    df_summary = pd.DataFrame([ac_940_result,
                               ac_970_result,
                               ac_1200_result,
                               dc_940nm,
                               dc_970nm,
                               dc_1200nm,
                               HeatRate_940,
                               HeatRate_970,
                               HeatRate_1200],
                              index=['940 AC',
                                     '970 AC',
                                     '1200 AC',
                                     '940 DC',
                                     '970 DC',
                                     '1200 DC',
                                     '940 HeartRate',
                                     '970 HeartRate',
                                     '1200 HeartRate'],
                              columns = [f] )

    #Save the new result into new Excel
    writer = ExcelWriter(f+'_new'+'.xlsx')
    df.to_excel(writer,'FFT Processed',index=False)
    df_peak_940.to_excel(writer,'940nm Peak',index=False)
    df_peak_970.to_excel(writer,'970nm Peak',index=False)
    df_peak_1200.to_excel(writer,'1200nm Peak',index=False)
    df_summary.to_excel(writer,'Summary',index=True)
    writer.save()
    print ('fileName:',f,'\n')    
        
def all_files_FFT():
    #創建待改的 txt list
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    txt_file = [f for f in files if f[-3:] == 'txt']
    #批次處理每個檔案
    for f in txt_file:
        file_FFT(f)

def single_file_FFT():
    x = filename_txt.get()
    f = str(x)+str('.txt')
    file_FFT(f)
    exit()

def set_file_name():
    x = e.get()
    filename_txt.set(str(x))

# 窗口主体框架 

window = tk.Tk()
window.title('FFT window')
window.geometry('350x300')

filename_txt = tk.StringVar()    # 这时文字变量储存器
filename_txt.set('檔案名稱')

# 設置Label
l1 = tk.Label(window, text='Step1:所有檔案轉檔')
l1.place(x=10, y= 10)

b1 = tk.Button(window,text="all_files_transfer",command=all_files_transfer)
b1.place(x=150, y=10)

l2 = tk.Label(window, text='Step2-1:所有檔案進行傅立葉轉換')
l2.place(x=10, y=50)

b2 = tk.Button(window,text="all_files_FFT",command=all_files_FFT)
b2.place(x=200, y=50)

l3 = tk.Label(window, text='單一檔案轉檔', bg='yellow')
l3.place(x=10, y=90)

l4 = tk.Label(window, text='Step2-2 輸入:')
l4.place(x=10, y=130)

e = tk.Entry(window, text = '檔案名稱')
e.place(x=90, y=130)

b3 = tk.Button(window,text="set_file_name",command=set_file_name)
b3.place(x=250,y=125)

l5 = tk.Label(window, text='檔名:')
l5.place(x=10, y=170)

l6 = tk.Label(window, textvariable=filename_txt)
l6.place(x=50,y=170)

b4 = tk.Button(window,text="single_file_FFT",command=single_file_FFT)
b4.place(x=10,y=210)


##显示出来
window.mainloop()