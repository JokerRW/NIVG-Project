# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:53:18 2018

@author: RICHARD.WENG
"""
#導入函數庫
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#----------------------需要設定的參數----------------------------------------------
#檔名參數
file_name='ANN_training_data_AC_combination_0122'
#輸入特徵參數
features_Num = 38 #要取的column數，也就是輸入變數
#輸入訓練次數
iteration = 30000
#輸入隱藏層特徵數
hiddenlayer_features = 5
#輸入學習速率
learning_rate = 0.3
#學習狀況記錄
lossbuck_train = []
lossbuck_test = []
#----------------------需要設定的參數----------------------------------------------

#---------------------ANN前數據讀檔處理---------------------------------------------
#讀檔創建Dataframe
#filepath='C:\\Users\\richard.weng\\Documents\\Python Scripts\\python_projects\\(1) NIVG Project\\ANN\\'
file_data = file_name+'.csv'
df0=pd.read_csv(file_data, encoding="gb18030")

#選擇受測人
#df0 = df0[df0.Name!='Moby'] #移除特定人
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

#df0 = df0.drop(columnDrop, axis=1)
#移除first column of tester
df = df0.iloc[:,1:]

#只取特定排做dataframe, 使用iloc的方式，若只寫一個column，會變成只有取值，不是變成dataframe
testNum = len(df)# 要取的測試組數，也就是測試的次數
dataFrame_x = df.iloc[0:testNum,1:features_Num+1]
#dataFrame_x_drop1300 = df.iloc[0:testNum,1:features_Num+1].drop(labels=["1300nm","1300nm.1"], axis="columns")
print(dataFrame_x.tail())
print('---------------------------------------------------------------')
dataFrame_y = df.iloc[0:testNum,0:1]
print(dataFrame_y.tail())
print('---------------------------------------------------------------')

#轉成array
row_data_x = np.array(dataFrame_x)
row_data_y = np.array(dataFrame_y)
print(np.shape(row_data_x))
print(np.shape(row_data_y))
print('---------------------------------------------------------------')

# 假設70%訓練，30%要驗證 (TrainingData and TestingData)
x_train, x_test, y_train, y_test = train_test_split(row_data_x,row_data_y,test_size=0.3,random_state= 25)
print(np.shape(x_train))
print(np.shape(y_train))

# scale units (Normalization)
x_train_normalized = x_train/np.amax(x_train, axis=0) # maximum of x_train array
x_test_normalized = x_test/np.amax(x_train, axis=0) # maximum of xPredicted (our input data for the prediction)
x_all_normalized = row_data_x/np.amax(x_train, axis=0) # maximum of xAll (our input data for the prediction)
y_train_normalized = y_train/600 # max test score is 600
y_test_normalized = y_test/600 # max test score is 600
#y_all_normalized = row_data_y/600 # max test score is 600
#---------------------ANN前數據讀檔處理---------------------------------------------

#----------------------------ANN 主程式---------------------------------------------
class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = features_Num 
        self.outputSize = 1
        self.hiddenSize = hiddenlayer_features 
        self.leaningRate = learning_rate 
        self.trainningTime = iteration 
        self.lost = 1e-20 #損失目標,初步隨便寫
    
    #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (12x13) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (13x1) weight matrix from hidden to output layer
        self.b1 = np.zeros((1, self.hiddenSize))
        self.b2 = np.zeros((1, self.outputSize))
    
    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1)+self.b1 # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2)+self.b2 # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o
    
    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propagate through the network
    
        # 輸出層誤差 o (error in output)
        self.o_error = y - o
        # 輸出層誤差度(applying derivative of sigmoid to o error)
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        
        # 往前輸出層誤差 z2 (後輸出層誤差度"內積"後層權重W2):# z2 error: how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.W2.T)
        # 輸出層誤差度(applying derivative of sigmoid to z2 error)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        
        self.db1 = np.sum(self.z2_delta, axis=0, keepdims=True)
        self.db2 = np.sum(self.o_delta, axis=0)
        
        #調整的原理: W的影響為"輸入參數"內積"輸出層誤差度*(例如X為輸入參數-->W-->Z2 (W-->Z之間為輸出層誤差度) 
        self.W1 += self.leaningRate*X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.leaningRate*self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
        self.b1 += self.leaningRate*self.db1
        self.b2 += self.leaningRate*self.db2
 
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self):
        np.savetxt('w1.txt', self.W1, fmt="%s")
        np.savetxt('w2.txt', self.W2, fmt="%s")
        np.savetxt('b1.txt', self.b1, fmt="%s")
        np.savetxt('b2.txt', self.b2, fmt="%s")
    
    def saveWeights_excel(self):
        df_W1 = pd.DataFrame(NN.W1)
        df_W2 = pd.DataFrame(NN.W2)
        df_b1 = pd.DataFrame(NN.b1)
        df_b2 = pd.DataFrame(NN.b2)
        writer = pd.ExcelWriter(file_name+'_GD_parameter'+'.xlsx')
        df.to_excel(writer,'Raw data',index=False)
        df_W1.to_excel(writer,'W1',index=False)
        df_W2.to_excel(writer,'W2',index=False)
        df_b1.to_excel(writer,'b1',index=False)
        df_b2.to_excel(writer,'b2',index=False)
        writer.save()     
       
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(x_test_normalized))
        print ("Output: \n" + str(self.forward(x_test_normalized)))
#----------------------------ANN 主程式---------------------------------------------

#----------------------------啟動ANN並執行------------------------------------------
NN = Neural_Network()
#開始訓練
for i in range(NN.trainningTime): # trains the NN times
    NN.train(x_train_normalized, y_train_normalized)
    if np.mean(np.square(y_train_normalized - NN.forward(x_train_normalized)))>NN.lost:
        if i % 10 == 0:
            lossbuck_train.append(float(np.mean(np.square(y_train_normalized - NN.forward(x_train_normalized)))))
            lossbuck_test.append(float(np.mean(np.square(y_test_normalized - NN.forward(x_test_normalized)))))
        if i % 1000 == 0:
            print ("# " + str(i) + "\n")
            #print ("Input (scaled): \n" + str(x_train_normalized)) #暫時不看畫面比較簡單
            print ("Actual Output: \n" + str(y_train_normalized*600))
            print ("Predicted Output: \n" + str(NN.forward(x_train_normalized)*600))
            print ("Loss: \n" + str(np.mean(np.square(y_train_normalized - NN.forward(x_train_normalized))))) # mean sum squared loss
            #畫圖
            y_test_predict=NN.forward(x_test_normalized)
            y_train_predict=NN.forward(x_train_normalized)
            plt.scatter(y_train, y_train_predict*600, label='Train sets (70% of data)')
            plt.scatter(y_test, y_test_predict*600, label='Verify sets (30% of data)')
            plt.title('ANN Simulation Result')
            plt.xlabel('Input glucose (mg/dL)')
            plt.ylabel('Predicted glucose (mg/dL)')
            plt.legend()
            plt.grid()
            plt.show()
            print('---------------------------------------------------------------')
    
    else:
        break

# NN.saveWeights()
NN.saveWeights_excel()
#NN.predict() #暫時不用看此輸入狀況
#print('---------------------------------------------------------------')
#----------------------------啟動ANN並執行------------------------------------------

#----------------------------確認執行後的結果------------------------------------------
# Fianl Result 圖       
y_test_predict=NN.forward(x_test_normalized)
y_train_predict=NN.forward(x_train_normalized)
plt.scatter(y_train, y_train_predict*600, label='Train sets (70% of data)')
plt.scatter(y_test, y_test_predict*600, label='Validation sets (30% of data)')
plt.title('Final ANN Simulation Result')
plt.xlabel('Input glucose (mg/dL)')
plt.ylabel('Predicted glucose (mg/dL)')
plt.legend()
plt.grid()
plt.show()
print ("Loss: \n" + str(np.mean(np.square(y_train_normalized - NN.forward(x_train_normalized)))))
print('---------------------------------------------------------------')
print ('待測糖值:\n',y_test)
print ('預測糖值:\n',y_test_predict*600)
print('---------------------------------------------------------------')

#Check final correlation
y_all_predict = NN.forward(x_all_normalized)*600
plt.scatter(row_data_y.flatten(),y_all_predict.flatten())
Name = df0['Name'].values.tolist()
df_result = pd.DataFrame({'Name': Name,'total_y': row_data_y.flatten(), 'total_pre_y': y_all_predict.flatten()})
print('相關性分析:\n',df_result.corr())
#列印出多少數據
print('總共樣本數:',len(df_result))

#觀察是否收斂
plt.figure()
# 在绘制时设置lable, 逗号是必须的
l1, = plt.plot(range(len(lossbuck_train)), lossbuck_train, label = 'train', linewidth = 2.0)
l2, = plt.plot(range(len(lossbuck_test)), lossbuck_test, label = 'test', color = 'red', linewidth = 2.0, linestyle = '--')
# 设置坐标轴的lable
plt.title('Loss Evaluation')
plt.xlabel('Steps (10 try/per step)')
plt.ylabel('Loss')
# 设置legend
plt.legend(handles = [l1, l2,], labels = ['train', 'test'], loc = 'best')
plt.show()

#觀察權重的強度
df_W1 = pd.DataFrame(NN.W1)
df_W2 = pd.DataFrame(NN.W2)
df_b1 = pd.DataFrame(NN.b1.flatten())
df_b2 = pd.DataFrame(NN.b2.flatten())
            
#畫成一個圖4X$
plt.figure(figsize=(10,8))
plt.subplot(221)
plt.plot(NN.W1)
plt.title('W1 Weight',loc='left')
plt.xlabel('Input features')
plt.ylabel('Value')
plt.subplot(222)
plt.plot(NN.W2)
plt.title('W2 Weight',loc='left')
plt.xlabel('Input features')
plt.ylabel('Value')
plt.subplot(223)
plt.plot(NN.b1.flatten())
plt.title('b1 Weight',loc='left')
plt.xlabel('Input features')
plt.ylabel('Value')
plt.subplot(224)
plt.scatter(1,NN.b2.flatten(),)
plt.title('b2 Weight',loc='left')
plt.xlabel('Input features')
plt.ylabel('Value')
plt.show()
print('---------------------------------------------------------------')
#----------------------------確認執行後的結果------------------------------------------

#可视化数据 input before normalized
## change the size of plot
# plt.rcParams["figure.figsize"] = (20,10)
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(df.iloc[:,1:9])
plt.title('Input before Nomalization')
plt.xlabel('Test sample Lable')
plt.ylabel('Input value (mV)')
#正規化後的input
plt.subplot(122)
plt.plot(x_all_normalized)
plt.title('Input after Nomalization')
plt.xlabel('Test sample Lable')
plt.ylabel('Input value (mV)')
plt.show()

#列印出多少數據
print('總共樣本數:',len(df_result))
#Save the new result into new csv
df_result.to_csv(file_name+'_GD_result'+'.csv')
#----------------------------確認執行後的結果------------------------------------------
