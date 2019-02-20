# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:29:49 2019

@author: RICHARD.WENG
"""

#導入函數庫
import numpy as np
import pandas as pd
import pyrenn as prn

#load NN method for feature work
LM_para_file = 'ANN_training_data_AC_combination_0122_LM_parameter.csv'
net = prn.loadNN(LM_para_file)

#Nomalized parameter
LM_para_normalfile = 'ANN_training_data_AC_combination_0122_LM_normalize_index.csv'
df_normalize = pd.read_csv(LM_para_normalfile,header = None)
normalize_factor = np.array(df_normalize.iloc[:,0].tolist())

#Read file
fileName = '0905-1_Volts_20190122_090921.txt_new.xlsx'
df_test = pd.read_excel(fileName,'Summary').T

#移除不需要的參數
columnDrop = [
              '1300 AC',
              '1300 DC',
              '1300 HR',
              '1300 Area',
              '1300 PWTT',
              '1300 BVI value',
              '1300 BVI amp',
              '1300 BVI time',
              '1300 BVA value',
             ]

df_test = df_test.drop(columnDrop, axis=1)
input_x = df_test.iloc[0,:].tolist()

#Diabetes index
input_x.extend([1.0,4.0])
input_f = np.array(input_x)
#參數正規化
input_x_normalized = input_f/normalize_factor
input_x_pre = input_x_normalized.reshape(29,1)


#輸出讀值
glucose = prn.NNOut(input_x_pre,net)*600
print(glucose)
