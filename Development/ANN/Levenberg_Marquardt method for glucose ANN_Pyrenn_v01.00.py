# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:45:43 2018

@author: RICHARD.WENG
"""

#導入函數庫
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrenn as prn
from sklearn.model_selection import train_test_split

#----------------------需要設定的參數----------------------------------------------
#檔名參數
file_name = 'ANN_training_Newjig_0213_all'
#輸入特徵參數
features_Num = 29
#輸入訓練次數
iteration = 15
#輸入第一隱藏層特徵數
hiddenlayer1_features = 5
#輸入第二隱藏層特徵數
hiddenlayer2_features = 1
#----------------------需要設定的參數----------------------------------------------

#------------------------------讀檔創建Dataframe---------------------------------
#filepath='C:\\Users\\richard.weng\\Documents\\Python Scripts\\python_projects\\(1) NIVG Project\\ANN\\'
file_data = file_name+'.csv'
df0 = pd.read_csv(file_data, encoding="gb18030")

#移除不需要的參數
#columnDrop = [
#              '940nm_AC','970nm_AC','1200nm_AC','1300nm_AC',
#              '940nm_DC','970nm_DC','1200nm_DC','1300nm_DC',
#              '940nm_HR','970nm_HR','1200nm_HR','1300nm_HR',
#              '940nm_Area','970nm_Area','1200nm_Area','1300nm_Area',
#              '940nm_PWTT','970nm_PWTT','1200nm_PWTT','1300nm_PWTT',
#              '940nm_BVI_Value','970nm_BVI_Value','1200nm_BVI_Value','1300nm_BVI_Value',
#              '940nm_BVI_amp','970nm_BVI_amp','1200nm_BVI_amp','1300nm_BVI_amp',
#              '940nm_BVI_time','970nm_BVI_time','1200nm_BVI_time','1300nm_BVI_time',
#              '940nm_BVA_value','970nm_BVA_value','1200nm_BVA_value','1300nm_BVA_value',
#              'DL_AC',
#              'DL_PC',
 #            ]

#移除不需要的參數
columnDrop = [
              '1300nm_AC',
              '1300nm_DC',
              '1300nm_HR',
              '1300nm_Area',
              '1300nm_PWTT',
              '1300nm_BVI_Value',
              '1300nm_BVI_amp',
              '1300nm_BVI_time',
              '1300nm_BVA_value',
             ]


df0 = df0.drop(columnDrop, axis=1)

#選擇受測人
#df = df[df.Name=='Nick']
df = df0.iloc[:,1:] #移除first column of tester

print (df.T.tail())
print ('--------------------------------------------')
print ('df 長度為:',len(df))
print ('--------------------------------------------')
P = df.T.iloc[1:features_Num+1,0:len(df)]
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
normalization_factor = np.amax(x_train, axis=0)
x_of_train = (x_train/normalization_factor).T
x_of_test = (x_test/normalization_factor).T
y_of_train = y_train.T/600
y_of_test = y_test.T/600
#------------------------------讀檔創建Dataframe---------------------------------

#----------------------------ANN 主程式---------------------------------------------
#8 input,2 hidden layer, 3 neuron (create NN)
net = prn.CreateNN([features_Num,hiddenlayer1_features,hiddenlayer2_features,1])
# Train by NN
net = prn.train_LM(x_of_train,y_of_train,net,verbose=True,k_max=iteration,E_stop=1e-10)
# print out result
y_prn_train = prn.NNOut(x_of_train,net)
y_prn_test = prn.NNOut(x_of_test,net)
# print('x train data 預測的 Predicted Y:','\n',y_prn_train*600)
# print('x test data 預測的 Predicted Y:','\n',y_prn_test*600)
#----------------------------ANN 主程式---------------------------------------------

#----------------------------確認執行後的結果------------------------------------------
# visualize result
plt.scatter(y_of_train*600, y_prn_train*600, label='Train sets (70% of data)')
plt.scatter(y_of_test*600, y_prn_test*600, label='Verify sets (30% of data)')
plt.title('ANN Simulation Result')
plt.xlabel('Input glucose (mg/dL)')
plt.ylabel('Predicted glucose (mg/dL)')
plt.legend()
plt.grid()
plt.show()
print ('測試組原本的糖值:','\n',y_of_test*600)
print ('測試組預測的糖值:','\n',y_prn_test*600)
#----------------------------確認執行後的結果------------------------------------------

#Save ANN
prn.saveNN(net, file_name+'_LM_parameter'+'.csv')

#----------------------------確認執行後的結果------------------------------------------
#Check final correlation
y_all = prn.NNOut((P.T/np.amax(x_train, axis=0)).T,net)*600
plt.scatter(Y.flatten(),y_all)
Name = df0['Name'].values.tolist()
df_result = pd.DataFrame({'Name': Name,'total_y': Y.flatten(), 'total_pre_y': y_all})
print('相關性分析:\n',df_result.corr())
#列印出多少數據
print('總共樣本數:',len(df_result))

#Normalized index
df_index = pd.Series(normalization_factor)
#Save the new result into new Excel
df_result.to_csv(file_name+'_LM_result'+'.csv')
df_index.to_csv(file_name+'_LM_normalize_index'+'.csv', index = False)

#load NN method for feature work
#net = prn.loadNN('LM_glucose_test.csv')
#----------------------------確認執行後的結果------------------------------------------

