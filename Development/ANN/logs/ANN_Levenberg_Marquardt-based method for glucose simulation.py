# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:15:35 2018

@author: RICHARD.WENG
"""

#導入函數庫
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrenn as prn
from sklearn.model_selection import train_test_split

#read All dataframe df from ANN_training_data.CSV
file_name='ANN_training_data_AC_combination_1129'
filepath='C:\\Users\\richard.weng\\Documents\\Python Scripts\\python_projects\\(1) NIVG Project\\ANN\\'
file_data = filepath+file_name+'.csv'
df0 = pd.read_csv(file_data)

#選擇受測人
#df = df[df.Name=='Nick']
#移除first column of tester
df = df0.iloc[:,1:]

print (df.T.tail())
print ('--------------------------------------------')
print ('df 長度為:',len(df))
print ('--------------------------------------------')
P = df.T.iloc[1:9,0:len(df)]
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

#8 input,2 hidden layer, 3 neuron (create NN)
net = prn.CreateNN([8,3,3,1])
# Train by NN
net = prn.train_LM(x_of_train,y_of_train,net,verbose=True,k_max=100,E_stop=1e-10)
# print out result
y_prn_train = prn.NNOut(x_of_train,net)
y_prn_test = prn.NNOut(x_of_test,net)
# print('x train data 預測的 Predicted Y:','\n',y_prn_train*600)
# print('x test data 預測的 Predicted Y:','\n',y_prn_test*600)
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
#Save ANN
prn.saveNN(net, file_name+'_LM_parameter'+'.csv')

#Check final correlation
y_all = prn.NNOut((P.T/np.amax(x_train, axis=0)).T,net)*600
plt.scatter(Y.flatten(),y_all)
Name = df0['Name'].values.tolist()
df_result = pd.DataFrame({'Name': Name,'total_y': Y.flatten(), 'total_pre_y': y_all})
print('相關性分析:\n',df_result.corr())
#列印出多少數據
print('總共樣本數:',len(df_result))
#Save the new result into new Excel
df_result.to_csv(file_name+'_LM_result'+'.csv')

#load NN method for feature work
#net = prn.loadNN('LM_glucose_test.csv')
