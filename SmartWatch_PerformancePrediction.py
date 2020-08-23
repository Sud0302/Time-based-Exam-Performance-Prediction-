# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:05:15 2020

@author: sudha
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression

dfx_train = pd.read_csv('D:\AI MAFIA\Assignment 1\Training Data\Linear_X_Train.csv')
dfy_train = pd.read_csv('D:\AI MAFIA\Assignment 1\Training Data\Linear_Y_Train.csv')
dfx_test = pd.read_csv('D:\AI MAFIA\Assignment 1\Testing Data\Linear_X_Test.csv')
dfx_train = dfx_train.values
dfy_train = dfy_train.values
dfx_test = dfx_test.values

x_train= dfx_train.reshape((-1,1))
y_train= dfy_train.reshape((-1,1))
x_test= dfx_test.reshape((-1,1))

plt.scatter(x_train,y_train)
model = LinearRegression()

model.fit(x_train,y_train)
output = model.predict(x_test)
bias = model.intercept_
coeff = model.coef_
print('Bias = ', bias)
print('Coefficient = ', coeff)
print('Accuracy = ',model.score(x_train,y_train))

plt.scatter(x_train,y_train,label='data')

plt.plot(x_test,output,color='black',label='prediction')
plt.ylabel('Feature')
plt.xlabel('Time')
plt.legend()
plt.show()
