import numpy as np
import scipy.io as sio
import glob
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import cycle
#import pystan
from scipy.optimize import curve_fit
from scipy import stats
import time
import matplotlib.patches as mpatches
from matplotlib import cm

color_cycle = cycle(['g', 'b', 'c', 'm', 'y', 'k'])



def fit_psych_curve_fast(data):
    # fit least square regression model

    def func(c,y0,a,c0,b):
        return y0+a/(1+np.exp(-1*(c-c0)/b))

    x_data = data.index.values
    y_data = data.values

    if len(y_data.shape)==1:
        y_data = np.expand_dims(y_data,1)


    #m_best = fit(m_init, x[index], y[index])
    print(y_data.shape, x_data.shape)

    fits = []

    for i in range(y_data.shape[1]):
        index = ~(np.isnan(x_data) | np.isnan(y_data[:, i]))
        popt, pcov = curve_fit(func,x_data[index],y_data[index,i], p0 = [ 0.03266858,  0.99823339, -0.10897152,  0.17083184], bounds = ((0.0,0.0,-1.0,-10),(1.0,1.0,1.0,10)), method = 'trf')
        fits.append(popt)

    #print(fits)
    return fits

def fit_psych_curve_fast2(x_data, y_data):
    # fit least square regression model

    def func(c,y0,a,c0,b):
        return y0+a/(1+np.exp(-1*(c-c0)/b))

    #x_data = data.index.values
    #y_data = data.values
    x_data = x_data[:, 0]
    if len(y_data.shape)==1:
        y_data = np.expand_dims(y_data,1)

    print(func(x_data, 0.03266858,  0.99823339, -0.10897152,  0.17083184 ).shape)
    #m_best = fit(m_init, x[index], y[index])

    print(y_data.shape, x_data.shape)

    fits = []

    #for i in range(y_data.shape[1]):
        #index = ~(np.isnan(x_data) | np.isnan(y_data[:, i]))
    popt, pcov = curve_fit(func,x_data,y_data[:, 0] , p0 = [ 0.03266858,  0.99823339, -0.10897152,  0.17083184], bounds = ((0.0,0.0,-1.0,-10),(1.0,1.0,1.0,10)), method = 'trf')
    fits.append(popt)

    #print(fits)
    return fits



def plot_fits(fits,color='k',alf=1):
    for i in range(len(fits)):
        plot_sigmoid(fits[i],color,alf)

def plot_sigmoid(p,color='k',alf=1.):
    x_val = np.linspace(-1,1,200).reshape(200,1)
    plt.plot(x_val,p[0]+p[1]/(1+np.exp(-1*(x_val-p[2])/p[3])),color,alpha=alf)
