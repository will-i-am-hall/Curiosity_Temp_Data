# -*- coding: utf-8 -*-
"""
Testing Machine Learning on Curiosity Rover Data Using scikit-learn 

Created on Tue Jul 25 18:20:27 2023

@author: Will Hall
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model, kernel_ridge, tree
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import mean_squared_error

df = pd.read_csv('REMS_Mars_Dataset.csv')

###############################################################################
#############################  Temp Ranges  ###################################
###############################################################################

df['ground_temp_range'] = np.abs(df['max_ground_temp(°C)'] - df['min_ground_temp(°C)'])
df['air_temp_range'] = np.abs(df['max_air_temp(°C)'] - df['min_air_temp(°C)'])

###############################################################################
############################  Data Cleaning  ##################################
###############################################################################

mngql = df['min_ground_temp(°C)'].quantile(0.01)
mngqu = df['min_ground_temp(°C)'].quantile(0.99) 
df = df[(df['min_ground_temp(°C)'] < mngqu) & (df['min_ground_temp(°C)'] > mngql)]

mxgql = df['max_ground_temp(°C)'].quantile(0.01)
mxgqu = df['max_ground_temp(°C)'].quantile(0.99) 
df = df[(df['max_ground_temp(°C)'] < mxgqu) & (df['max_ground_temp(°C)'] > mxgql)]

angql = df['min_air_temp(°C)'].quantile(0.01)
angqu = df['min_air_temp(°C)'].quantile(0.99) 
df = df[(df['min_air_temp(°C)'] < angqu) & (df['min_air_temp(°C)'] > angql)]

axgql = df['max_air_temp(°C)'].quantile(0.01)
axgqu = df['max_air_temp(°C)'].quantile(0.99) 
df = df[(df['max_air_temp(°C)'] < axgqu) & (df['max_air_temp(°C)'] > axgql)]

###############################################################################
###########################  Machine Learning  ################################
###############################################################################

feature_cols = ['sol_number', 'air_temp_range']
x = df.loc[:, feature_cols].values
y = df.loc[:, 'ground_temp_range'].values

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size = 0.9)

reg = linear_model.BayesianRidge(fit_intercept = False, n_iter = 5000)
reg.fit(train_x, train_y)
y_predict = reg.predict(test_x)

kernel =  kernels.ExpSineSquared() + kernels.RationalQuadratic() + kernels.DotProduct()
krr = kernel_ridge.KernelRidge(alpha = 50, kernel = kernel)
krr.fit(train_x, train_y)
y_predict2 = krr.predict(test_x)

gpr = GaussianProcessRegressor(kernel=kernel, alpha = 50, n_restarts_optimizer= 25).fit(train_x, train_y)
y_predict3 = gpr.predict(test_x)

clf = tree.DecisionTreeRegressor()
clf.fit(train_x, train_y)
y_predict4 = clf.predict(test_x)

###############################################################################
###############################   Scoring   ###################################
###############################################################################

test_sol = []
for i in test_x:
    isol = i[0]
    test_sol.append(isol)

test_coords = sorted(set(zip(test_sol, test_y)))
sortx, sorty = zip(*test_coords)

orderpredict1 = sorted(set(zip(test_sol, y_predict)))
sortx, sortpy1 = zip(*orderpredict1)
orderpredict2 = sorted(set(zip(test_sol, y_predict2)))
sortx, sortpy2 = zip(*orderpredict2)
orderpredict3 = sorted(set(zip(test_sol, y_predict3)))
sortx, sortpy3 = zip(*orderpredict3)
orderpredict4 = sorted(set(zip(test_sol, y_predict4)))
sortx, sortpy4 = zip(*orderpredict4)

sortx = list(sortx)
sorty = list(sorty)
sortpy1 = list(sortpy1)
sortpy2 = list(sortpy2)
sortpy3 = list(sortpy3)
sortpy4 = list(sortpy4)

acc1 = mean_squared_error(sorty, sortpy1)
acc2 = mean_squared_error(sorty, sortpy2)
acc3 = mean_squared_error(sorty, sortpy3)
acc4 = mean_squared_error(sorty, sortpy4)

print("--------------------------------------------")
print("|---------| Mean Squared Errors: |---------|")
print("|------------------------------------------|")
print("| Bayesian Ridge    : ", acc1, " |")
print("| Kernel Ridge      : ", acc2, " |")
print("| Gaussian Process  : ", acc3, " |")
print("| Decision Trees    : ", acc4, " |")
print("--------------------------------------------")

###############################################################################
###############################  Plotting  ####################################
###############################################################################

fig, axs = plt.subplots(2, 2)
fig.suptitle('Martian Ground Temp Range (data vs model)')
fig.tight_layout()


axs[0, 0].set_ylabel('Temperature Range(°C)')
axs[0, 0].set_xlabel('Time (Sol number)')
axs[0, 0].grid(True)
axs[0, 0].plot(sortx, sorty, '#666666', label = 'Real Data')
axs[0, 0].plot(sortx, sortpy1, '--r', label = 'Bayesian Ridge')
axs[0, 0].set_title('Mean Square Error: ' + str(acc1))
axs[0, 0].legend()

axs[0, 1].set_ylabel('Temperature Range(°C)')
axs[0, 1].set_xlabel('Time (Sol number)')
axs[0, 1].grid(True)
axs[0, 1].plot(sortx, sorty, '#666666', label = 'Real Data')
axs[0, 1].plot(sortx, sortpy2, '--g', label = 'Kernel Ridge')
axs[0, 1].set_title('Mean Square Error: ' + str(acc2))
axs[0, 1].legend()

axs[1, 0].set_ylabel('Temperature Range(°C)')
axs[1, 0].set_xlabel('Time (Sol number)')
axs[1, 0].grid(True)
axs[1, 0].plot(sortx, sorty, '#666666', label = 'Real Data')
axs[1, 0].plot(sortx, sortpy3,'--b', label = 'Gaussian Process')
axs[1, 0].set_title('Mean Square Error: ' + str(acc3))
axs[1, 0].legend()

axs[1, 1].set_ylabel('Temperature Range(°C)')
axs[1, 1].set_xlabel('Time (Sol number)')
axs[1, 1].grid(True)
axs[1, 1].plot(sortx, sorty, '#666666', label = 'Real Data')
axs[1, 1].plot(sortx, sortpy4, '--y', label = 'Decision Trees')
axs[1, 1].set_title('Mean Square Error: ' + str(acc4))
axs[1, 1].legend()

plt.show()

