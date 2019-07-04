#some functions to measure errors

import numpy as np
from scipy.stats import pearsonr
import sklearn


#x and y are two array which x[i] and y[i] are two arrays corresponding to the data data of audio signal. This function returns the average value of all the pearsons coefficient of those arrays 

def mse_audio(x,y):
    lenght=len(x)
    error=0
    if(lenght!=len(y)):
        raise ValueError("the two arrays must have the same lenght")
    for i in range(lenght):
        error=sklearn.metrics.mean_squared_error(np.reshape(x[i],(x.shape[1]))*1/np.linalg.norm(x[i]),np.reshape(y[i],(y.shape[1]))*1/np.linalg.norm(y[i]))+error
        
        i=i+1
    return error*1/lenght


def mse_melSpectro(x,y):
    lenght=len(x)
    error=0
    if(lenght!=len(y)):
        raise ValueError("the two arrays must have the same lenght")
    for i in range(lenght):
        error=sklearn.metrics.mean_squared_error(np.reshape(x[i],(x.shape[1],x.shape[2]))*1/np.linalg.norm(x[i]),np.reshape(y[i],(y.shape[1],y.shape[2]))*1/np.linalg.norm(y[i]))+error
        
        i=i+1
    return error*1/lenght