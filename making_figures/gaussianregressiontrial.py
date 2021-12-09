#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:07:33 2021

@author: adamfoster
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 1, 500)).T

# ----------------------------------------------------------------------
# now the noisy case
y_data_1 = np.array(([0.01569564, 0.01513832, 0.01492116, 0.0151135 , 0.01555074,
       0.01627925, 0.01752451, 0.01945679, 0.02252841, 0.02740485,
       0.0345932]))

y_data_2 = np.array(([0.01435578, 0.01427549, 0.01529258, 0.01796716, 0.02343567]))

list_y = [y_data_1, y_data_2]
# Observations and noise

#dy = 0.5 + 1.0 * np.random.random(y.shape)
x_data_1 = np.array(([0.08063873, 0.09676648, 0.11289422, 0.12902197, 0.14514971,
       0.16127746, 0.17740521, 0.19353296, 0.20966069, 0.22578844,
       0.24191618]))

x_data_2 = np.array(([0.13822635, 0.16587162, 0.19351688, 0.22116216, 0.24880742]))

list_x= [x_data_1, x_data_2]
plt.figure()
i=0
for index in range(len(list_y)):
    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-5, 1e5)) * RBF(10, (1e-5, 1e5))
    
    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    #gp.fit(np.atleast_2d(list_x[i]).T, list_y[i])
    gp.fit(np.atleast_2d(x_data_2).T, y_data_2)
    
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    
    plt.errorbar(x_data_2, y_data_2, fmt="r.", markersize=10, label="Observations", alpha=0.7)
    plt.plot(x, y_pred, "b-", label="Prediction", alpha=0.7)
    plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=0.5,
        fc="b",
        ec="None",
        label="95% confidence interval",
    )
    i+=1
    
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.ylim(0, 0.25)
plt.xlim(0, 0.25)
plt.legend(loc="upper left")

plt.show()

"""
plt.figure()
plt.errorbar(x_data, y_data, marker ='x', alpha=0.3)
plt.ylim(0, 1)
plt.xlim(0, 0.25)
plt.show()
"""
