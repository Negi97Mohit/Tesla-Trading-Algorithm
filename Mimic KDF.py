# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:28:09 2021

Author: Mohit Negi

Team: Tesla (Ticker Name: TSLA)

"""

from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import seaborn as sns; sns.set(color_codes=True)
import math


#Date range from 1st Jan 2020 to 31st May 2020
#plotting the close value graph
TSLA = DataReader('TSLA',  'yahoo', datetime(2020,1,1), datetime(2020,5,31))
dates =[]
for x in range(len(TSLA)):
    newdate = str(TSLA.index[x])
    newdate = newdate[0:10]
    dates.append(newdate)

TSLA['Date']=dates

Y = TSLA['Close'].values

#Kernal Density Plot
sns.kdeplot(data=Y)


"""dt=TSLA['Date'].values
plt.plot(dt,Y)
plt.xlabel('Date-Time')
plt.ylabel('Price (USD)')
"""



#we get a bi-node density.
# 1st Normal Distribution with 1st node
u = np.random.normal(0,1,1);
mu_1=102
Sigma1=90
r1 = mu_1 + math.sqrt(Sigma1)*u;
''' Visualize y_1 = f(x_1) '''
sigma_1 = math.sqrt(Sigma1);
x_1 = np.linspace(60, 130, 2000)
y_1 = ss.norm.pdf(x_1, mu_1, sigma_1)
plt.figure(2)
plt.plot(x_1, y_1)


# 1st Normal Distribution with 2nd node
u = np.random.normal(0,1,1);
mu_2=158
Sigma2=175
r1 = mu_1 + math.sqrt(Sigma1)*u;
''' Visualize y_1 = f(x_1) '''
sigma_2 = math.sqrt(Sigma2);
x_2 = np.linspace(130, 225, 2000)
y_2 = ss.norm.pdf(x_2, mu_2, sigma_2)
plt.figure(3)
plt.plot(x_2, y_2)


#Normal Mixture of two Normal Distribution
p = 0.4; 
S = 5000;
r = np.zeros(S);
y = np.zeros(S);
for s in range(1,S):
    eps = np.random.normal(0,1,1);
    r1 = mu_1 + math.sqrt(Sigma1)*eps;
    r2 = mu_2 + math.sqrt(Sigma2)*eps;
    u = np.random.uniform(0,1,1);
    r[s] = r1*(u<p)+r2*(u>=p);

plt.figure(4)
sns.kdeplot(r,shade=True, color="r");



plt.show()
