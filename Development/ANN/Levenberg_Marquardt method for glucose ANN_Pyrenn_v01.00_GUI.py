# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:24:09 2018

@author: RICHARD.WENG
"""

#導入函數庫
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrenn as prn
from sklearn.model_selection import train_test_split
import tkinter as tk

#In pyrenn the transferfunction is defined as:
#the hyperbolic tangent a=tanh(n) for all neurons in hidden layer
#the linear function y=a=n for all neurons in the output layer

# 窗口主体框架 
window = tk.Tk()
window.title('ANN window of Levenberg_Marquardt-based method')
window.geometry('400x400')

#----------------------需要設定的參數----------------------------------------------
#檔名參數
file_name = tk.StringVar()
file_name.set('ANN_training_data_AC_combination_1212')
#輸入特徵參數
features_Num = tk.IntVar()
features_Num.set(8)
#輸入訓練次數
iteration = tk.IntVar()
iteration.set(100)
#輸入第一隱藏層特徵數
hiddenlayer1_features = tk.IntVar()
hiddenlayer1_features.set(3)
#輸入第二隱藏層特徵數
hiddenlayer2_features = tk.IntVar()
hiddenlayer2_features.set(3)
#----------------------需要設定的參數------------------------------------------------

# 執行內容:  
def Tranning_by_Neural_Network():
    #------------------------------讀檔創建Dataframe---------------------------------
    #filepath='C:\\Users\\richard.weng\\Documents\\Python Scripts\\python_projects\\(1) NIVG Project\\ANN\\'
    file_data = file_name.get()+'.csv'
    df0=pd.read_csv(file_data)
    
    #選擇受測人
    #df = df[df.Name=='Nick']
    df = df0.iloc[:,1:] #移除first column of tester
    
    print (df.T.tail())
    print ('--------------------------------------------')
    print ('df 長度為:',len(df))
    print ('--------------------------------------------')
    P = df.T.iloc[1:features_Num.get()+1,0:len(df)]
    print(P.tail())
    print('input的格式:',P.shape)
    print ('--------------------------------------------')
    Y = df.T.iloc[0:1,0:len(df)]
    print(Y.tail())
    print('output的格式:',Y.shape)
    print ('--------------------------------------------')
    #轉成2d array
    P = np.array(P)
    Y = np.array(Y)

    # 假設70%訓練，30%要驗證 (TrainingData and TestingData)
    x_train, x_test, y_train, y_test = train_test_split(P.T,Y.T,test_size=0.3,random_state= None)
    x_of_train = (x_train/np.amax(x_train, axis=0)).T
    x_of_test = (x_test/np.amax(x_train, axis=0)).T
    y_of_train = y_train.T/600
    y_of_test = y_test.T/600
    #------------------------------讀檔創建Dataframe------------------------------------
    
    #----------------------------ANN 主程式---------------------------------------------
    #8 input,2 hidden layer, 3 neuron (create NN)
    net = prn.CreateNN([features_Num.get(),hiddenlayer1_features.get(),hiddenlayer2_features.get(),1])
    # Train by NN
    net = prn.train_LM(x_of_train,y_of_train,net,verbose=True,k_max=iteration.get(),E_stop=1e-10)
    # print out result
    y_prn_train = prn.NNOut(x_of_train,net)
    y_prn_test = prn.NNOut(x_of_test,net)
    # print('x train data 預測的 Predicted Y:','\n',y_prn_train*600)
    # print('x test data 預測的 Predicted Y:','\n',y_prn_test*600)
    #----------------------------ANN 主程式---------------------------------------------
    
    #----------------------------確認執行後的結果------------------------------------------
    # visualize result
    plt.scatter(y_of_train*600, y_prn_train*600)
    plt.scatter(y_of_test*600, y_prn_test*600)
    plt.title('ANN Simulation Result')
    plt.xlabel('Input glucose (mg/dL)')
    plt.ylabel('Predicted glucose (mg/dL)')
    plt.grid()
    plt.show()
    print ('測試組原本的糖值:','\n',y_of_test*600)
    print ('測試組預測的糖值:','\n',y_prn_test*600)
    #----------------------------確認執行後的結果------------------------------------------
    
    #Save ANN
    prn.saveNN(net, file_name.get()+'_LM_parameter'+'.csv')

    #Check final correlation
    y_all = prn.NNOut((P.T/np.amax(x_train, axis=0)).T,net)*600
    plt.scatter(Y.flatten(),y_all)
    Name = df0['Name'].values.tolist()
    df_result = pd.DataFrame({'Name': Name,'total_y': Y.flatten(), 'total_pre_y': y_all})
    print('相關性分析:\n',df_result.corr())
    #列印出多少數據
    print('總共樣本數:',len(df_result))
    #Save the new result into new Excel
    df_result.to_csv(file_name.get()+'_LM_result'+'.csv')

    #load NN method for feature work
    #net = prn.loadNN('LM_glucose_test.csv')

#----------------------窗口相關的item------------------------
# 設計參數修改介面: 檔名修改
def set_file_name():
    x = e1.get()
    file_name.set(str(x))

l1 = tk.Label(window, text='Step1:輸入檔案名稱')
l1.place(x=10, y= 10)

e1 = tk.Entry(window, text = '檔案名稱')
e1.place(x=150, y=10)

b1 = tk.Button(window,text="set_file_name",command=set_file_name)
b1.place(x=300, y=6)

l2 = tk.Label(window, text='檔名:')
l2.place(x=10, y=40)

l3 = tk.Label(window, textvariable=file_name)
l3.place(x=50, y=40)

# 設計參數修改介面: 特徵數
def set_feature_num():
    x = e2.get()
    features_Num.set(int(x))

l4 = tk.Label(window, text='Step2:特徵數')
l4.place(x=10, y= 70)

l5 = tk.Label(window, textvariable=str(features_Num))
l5.place(x=140, y= 70)

e2 = tk.Entry(window, text = '特徵數')
e2.place(x=190, y=70, width=50)

b2 = tk.Button(window,text="set_feature_num",command=set_feature_num)
b2.place(x=250, y=66)

# 設計參數修改介面: 訓練次數
def set_tranning_num():
    x = e3.get()
    iteration.set(int(x))

l6 = tk.Label(window, text='Step3:訓練數')
l6.place(x=10, y= 100)

l7 = tk.Label(window, textvariable=str(iteration))
l7.place(x=140, y= 100)

e3 = tk.Entry(window, text = '訓練次數')
e3.place(x=190, y=100, width=50)

b3 = tk.Button(window,text="set_tranning_num",command=set_tranning_num)
b3.place(x=250, y=96)

# 設計參數修改介面: 隱藏層1特徵數
def set_hiddenlayer1_features_num():
    x = e4.get()
    hiddenlayer1_features.set(int(x))

l8 = tk.Label(window, text='Step4:隱藏"1"特徵數')
l8.place(x=10, y= 130)

l9 = tk.Label(window, textvariable=str(hiddenlayer1_features))
l9.place(x=140, y= 130)

e4 = tk.Entry(window, text = '隱藏層"1"特徵數')
e4.place(x=190, y=130, width=50)

b4 = tk.Button(window,text="set_hidden1_num",command=set_hiddenlayer1_features_num)
b4.place(x=250, y=126)

# 設計參數修改介面: 隱藏層2特徵數
def set_hiddenlayer2_features_num():
    x = e5.get()
    hiddenlayer2_features.set(int(x))

l10 = tk.Label(window, text='Step5:隱藏"2"特徵數')
l10.place(x=10, y= 160)

l11 = tk.Label(window, textvariable=str(hiddenlayer2_features))
l11.place(x=140, y= 160)

e5 = tk.Entry(window, text = '隱藏層"2"特徵數')
e5.place(x=190, y=160, width=50)

b5 = tk.Button(window,text='set_hidden2_num',command=set_hiddenlayer2_features_num)
b5.place(x=250, y=156)

# 執行dataframe 生成
# b6 = tk.Button(window,text="Create Data for analysis", bg='green',command=dataframe_creating)
# b6.place(x=190, y=220)

# 執行ANN
b7 = tk.Button(window,text="Run_ANN", bg='yellow',command=Tranning_by_Neural_Network)
b7.place(x=190, y=250)

#----------------------窗口相關的item------------------------    

##顯示視窗
window.mainloop()