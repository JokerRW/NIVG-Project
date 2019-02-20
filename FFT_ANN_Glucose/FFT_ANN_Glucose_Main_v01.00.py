# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:17:17 2019

@author: RICHARD.WENG"""

import os
import tkinter as tk
from tkinter import Scrollbar

# 窗口主体框架 
window = tk.Tk()
window.title('NIVG_LM_29_Prediction')
window.geometry('700x450')
scrollbar = Scrollbar(window)
scrollbar.pack(side='right', fill='y')

#----------------------需要設定的參數----------------------------------------------
#待測檔名及初始化
file_name = tk.StringVar()
file_name.set('0905-1_Volts_20190122_090921')

#LM參數及初始化
LM_para_name = tk.StringVar()
LM_para_name.set('ANN_training_Newjig_0213_all_LM_parameter')

#參數正規化參數及初始化
LM_para_normal_name = tk.StringVar()
LM_para_normal_name.set('ANN_training_Newjig_0213_all_LM_normalize_index')

#Diabetes index 參數及初始化
Diabete_index1 = tk.IntVar()
Diabete_index1.set(1)

Diabete_index2 = tk.IntVar()
Diabete_index2.set(4)

#Glucose prediction result
Glucose_predic = tk.DoubleVar()

#FFT process status
FFT_status = tk.StringVar()
#----------------------需要設定的參數----------------------------------------------

#-------------------------原始檔案的副檔名xls錯誤，所以需要重新命名txt-------------------------
def Rename_file_command():
    from NIVG_Cal import FFT_v01
    #將所有檔名更換成txt
    FFT_v01.all_files_transfer()
#-------------------------原始檔案的副檔名xls錯誤，所以需要重新命名txt-------------------------

#------------------------------------FFT 處理 -------------------------  
def all_FFT_command():
    from NIVG_Cal import FFT_v01
    #將所有檔案FFT處理，並且進行檔案存取
    FFT_v01.all_files_FFT()

def single_file_FFT_command():
    from NIVG_Cal import FFT_v01
    #將單一檔案FFT處理的模式
    Test_file = os.path.abspath('Raw data')+'\\'+file_name.get()+'.txt'
    FFT_v01.file_FFT(Test_file)
    FFT_status.set('Done')
#------------------------------------FFT 處理 -------------------------
    
#------------------------------------LM_ANN_Predcion-------------------------
def glucose_prediction_command():
    #參數檔案位置提供
    LM_para_file = os.path.abspath('Parameter')+'\\'+LM_para_name.get()+'.csv'
    LM_para_normalfile = os.path.abspath('Parameter')+'\\'+LM_para_normal_name.get()+'.csv'
    Test_file = os.path.abspath('Raw data')+'\\'+file_name.get()+'.txt'
    File_for_predic = Test_file + '_new.xlsx'
    DI1= Diabete_index1.get()
    DI2= Diabete_index2.get()
    from NIVG_Cal import LM_ANN_v01
    Glu = LM_ANN_v01.LM_predic(LM_para_file,LM_para_normalfile,File_for_predic,DI1,DI2)
    Glucose_predic.set(Glu)
#------------------------------------LM_ANN_Predcion-------------------------

#----------------------窗口相關的item------------------------
# 設計檔案更名介面: 副檔名修改
l1 = tk.Label(window, text='Step 0:  檔案更名')
l1.place(x=10, y= 10)

b1 = tk.Button(window,text="File rename", command = Rename_file_command,bg='yellow')
b1.place(x=150, y=6)

b9 = tk.Button(window,text="All FFT", command = all_FFT_command,bg='gray')
b9.place(x=240, y=6)

# 設計參數修改介面: 檔名輸入
def set_file_name():
    x = lb1.get(lb1.curselection())
    file_name.set(str(x))

l2 = tk.Label(window, text='Step 1: 輸入檔案名稱')
l2.place(x=10, y= 40)

files1 = [f for f in os.listdir('Raw data') if os.path.isfile(os.path.abspath('Raw data')+'\\'+f)]
txt_file = []
for f in files1:
    if f[-3:] == 'txt':
        txt_file.append(f[:-4])

files_selection_txt = tk.StringVar()
files_selection_txt.set(txt_file)

lb1 = tk.Listbox(window,listvariable=files_selection_txt, width=45)
lb1.place(x=150, y=40)
# attach listbox to scrollbar
lb1.config(yscrollcommand=scrollbar.set, height = 1)
scrollbar.config(command=lb1.yview)

b2 = tk.Button(window,text="set_file_name",command=set_file_name)
b2.place(x=470, y=36)

b3 = tk.Button(window,text='FFT',command=single_file_FFT_command,bg='yellow')
b3.place(x=560, y=36)

l17 = tk.Label(window, textvariable= FFT_status,
               bg='green2', width=4, height=2)
l17.place(x=600, y=32)

l3 = tk.Label(window, text='檔名:')
l3.place(x=10, y=70)

l4 = tk.Label(window, textvariable=file_name)
l4.place(x=50, y=70)

# 設計參數修改介面: LM參數輸入
def set_LM_para_name():
    x = lb2.get(lb2.curselection())
    LM_para_name.set(str(x))

l5 = tk.Label(window, text='Step 2: 輸入LM參數檔')
l5.place(x=10, y= 100)

files2 = [f for f in os.listdir('Parameter') if os.path.isfile(os.path.abspath('Parameter')+'\\'+f)]
LM_para_file = []
for f in files2:
    if f[-13:] == 'parameter.csv':
        LM_para_file.append(f[:-4])

files_selection_csv = tk.StringVar()
files_selection_csv.set(LM_para_file)

lb2 = tk.Listbox(window,listvariable=files_selection_csv, width=45)
lb2.place(x=150, y=100)
# attach listbox to scrollbar
lb2.config(yscrollcommand=scrollbar.set, height = 1)
scrollbar.config(command=lb2.yview)

b4 = tk.Button(window,text="set_LM_para_name",command=set_LM_para_name)
b4.place(x=470, y= 96)

l6 = tk.Label(window, text='檔名:')
l6.place(x=10, y=130)

l7 = tk.Label(window, textvariable=LM_para_name)
l7.place(x=50, y=130)

# 設計參數修改介面: 正規化參數輸入
def set_LM_para_normal():
    x = lb3.get(lb3.curselection())
    LM_para_normal_name.set(str(x))

l8 = tk.Label(window, text='Step 3: 輸入正規化')
l8.place(x=10, y= 160)

files3 = [f for f in os.listdir('Parameter') if os.path.isfile(os.path.abspath('Parameter')+'\\'+f)]
Normalize_index_file = []
for f in files3:
    if f[-9:] == 'index.csv':
        Normalize_index_file.append(f[:-4])

files_selection_csv = tk.StringVar()
files_selection_csv.set(Normalize_index_file)

lb3 = tk.Listbox(window,listvariable=files_selection_csv, width=45)
lb3.place(x=150, y=160)
# attach listbox to scrollbar
lb3.config(yscrollcommand=scrollbar.set, height = 1)
scrollbar.config(command=lb3.yview)

b5 = tk.Button(window,text="set_LM_para_normal",command=set_LM_para_normal)
b5.place(x=470, y= 156)

l9 = tk.Label(window, text='檔名:')
l9.place(x=10, y=190)

l10 = tk.Label(window, textvariable=LM_para_normal_name)
l10.place(x=50, y=190)

# 設計參數修改介面: Diabetes index 參數
def set_Diabete_index1():
    x = e4.get()
    Diabete_index1.set(int(x))

l11 = tk.Label(window, text='Step 4: Diabete_index1')
l11.place(x=10, y= 220)

l12 = tk.Label(window, textvariable=str(Diabete_index1))
l12.place(x=180, y= 220)

e4 = tk.Entry(window, text = 'Diabete_index1')
e4.place(x=230, y=220, width=50)

b6 = tk.Button(window,text="set_Diabete_index1",command=set_Diabete_index1)
b6.place(x=300, y=216)

def set_Diabete_index2():
    x = e5.get()
    Diabete_index2.set(int(x))

l13 = tk.Label(window, text='Step 5: Diabete_index2')
l13.place(x=10, y= 250)

l14 = tk.Label(window, textvariable=str(Diabete_index2))
l14.place(x=180, y= 250)

e5 = tk.Entry(window, text = 'Diabete_index2')
e5.place(x=230, y=250, width=50)

b7 = tk.Button(window,text="set_Diabete_index2",command=set_Diabete_index2)
b7.place(x=300, y=246)

# 執行Prediction

b8 = tk.Button(window,text="Run_Prediction", bg='yellow',command=glucose_prediction_command)
b8.place(x=10, y=280)

l15 = tk.Label(window, textvariable=str(Glucose_predic),
               bg='azure', font=('Arial', 30), width=5, height=2)
l15.place(x=130, y=280)

l16 = tk.Label(window, text= 'mg/dL',font = ('Arial', 20))
l16.place (x=260, y=340)


##显示出来
window.mainloop()  
    
    