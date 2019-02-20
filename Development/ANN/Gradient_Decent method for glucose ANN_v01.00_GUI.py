# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:59:46 2018

@author: RICHARD.WENG
"""

#導入函數庫
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tkinter as tk

# 窗口主体框架 
window = tk.Tk()
window.title('ANN window of Gradient_Decent-based method')
window.geometry('400x400')

#----------------------需要設定的參數----------------------------------------------
#檔名參數
file_name = tk.StringVar()
file_name.set('ANN_training_data_AC_combination_1129')
#輸入特徵參數
features_Num = tk.IntVar()
features_Num.set(8)
#輸入訓練次數
iteration = tk.IntVar()
iteration.set(20000)
#輸入隱藏層特徵數
hiddenlayer_features = tk.IntVar()
hiddenlayer_features.set(5)
#輸入學習速率
learning_rate = tk.DoubleVar()
learning_rate.set(0.3)

#學習狀況記錄
lossbuck_train = []
lossbuck_test = []
#----------------------需要設定的參數------------------------------------------------

def Tranning_by_Neural_Network():
    #---------------------ANN前數據讀檔處理---------------------------------------------
    #讀檔創建Dataframe
    #filepath='C:\\Users\\richard.weng\\Documents\\Python Scripts\\python_projects\\(1) NIVG Project\\ANN\\'
    file_data = file_name.get()+'.csv'
    df0=pd.read_csv(file_data)
    
    #選擇受測人
    #df0 = df0[df0.Name!='Moby'] #移除特定人
    df = df0.iloc[:,1:] #移除first column of tester
    
    #只取特定排做dataframe, 使用iloc的方式，若只寫一個column，會變成只有取值，不是變成dataframe
    testNum = len(df) # 要取的測試組數，也就是測試的次數
    dataFrame_x = df.iloc[0:testNum,1:features_Num.get()+1]
    #dataFrame_x_drop1300 = df.iloc[0:testNum,1:columnNum+1].drop(labels=["1300nm","1300nm.1"], axis="columns")
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
    x_train, x_test, y_train, y_test = train_test_split(row_data_x,row_data_y,test_size=0.3,random_state=10)
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
            self.inputSize = features_Num.get() #根據input參數, 現在有12個
            self.outputSize = 1
            self.hiddenSize = hiddenlayer_features.get() #先比12個參數多1，討論是否太多
            self.leaningRate = learning_rate.get() #隨便設計的
            self.trainningTime = iteration.get() #訓練次數
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
            writer = pd.ExcelWriter(file_name.get()+'_GD_parameter'+'.xlsx')
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
    #根據欲調整之參數,修改初始參數
    NN.inputSize = features_Num.get() #根據input參數, 現在有12個
    NN.outputSize = 1
    NN.hiddenSize = hiddenlayer_features.get() #先比12個參數多1，討論是否太多
    NN.leaningRate = learning_rate.get() #隨便設計的
    NN.trainningTime = iteration.get() #訓練次數
    NN.lost = 1e-20 #損失目標,初步隨便寫  
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
                plt.scatter(y_test, y_test_predict*600, label='Test sets (30% of data)')
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
    plt.scatter(y_test, y_test_predict*600, label='Test sets (30% of data)')
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
    df_result.to_csv(file_name.get()+'_GD_result'+'.csv')
    #----------------------------確認執行後的結果------------------------------------------       
        
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

# 設計參數修改介面: 隱藏層特徵數
def set_hiddenlayer_features_num():
    x = e4.get()
    hiddenlayer_features.set(int(x))

l8 = tk.Label(window, text='Step4:隱藏層特徵數')
l8.place(x=10, y= 130)

l9 = tk.Label(window, textvariable=str(hiddenlayer_features))
l9.place(x=140, y= 130)

e4 = tk.Entry(window, text = '隱藏層特徵數')
e4.place(x=190, y=130, width=50)

b4 = tk.Button(window,text="set_hidden_num",command=set_hiddenlayer_features_num)
b4.place(x=250, y=126)

# 設計參數修改介面: 訓練速率
def set_learningRate():
    x = e5.get()
    learning_rate.set(float(x))

l10 = tk.Label(window, text='Step5:訓練速率')
l10.place(x=10, y= 160)

l11 = tk.Label(window, textvariable=str(learning_rate))
l11.place(x=140, y= 160)

e5 = tk.Entry(window, text = '訓練速率')
e5.place(x=190, y=160, width=50)

b5 = tk.Button(window,text="set_learningRate",command=set_learningRate)
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