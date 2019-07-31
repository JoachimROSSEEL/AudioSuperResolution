import numpy as np
import scipy.signal
from matplotlib.pyplot import *
import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
from filter_function import *
from spectrogram import *
from wav_preprocessing import *
import matplotlib.image as mpimg
import glob
from keras.models import Sequential,Model,load_model
from notebook_display_error import *
from error_measure import *

def display_mel_spectro(display_index,Xtest_mel,Xtest_fil_mel,Spec_audio_decoded,decoded_mel,Spec_lin):
    plt.figure(figsize=(14, 6))
    plt.subplot(1,2,1)
    librosa.display.specshow(np.reshape(Xtest_mel[display_index],(1024,16)), sr=8000, x_axis='time', y_axis='mel',fmax=4000,cmap='Spectral_r')
    title("expected mel spectrogram")
    colorbar()
    
    plt.subplot(1,2,2)
    librosa.display.specshow(np.reshape(Xtest_fil_mel[display_index],(1024,16)), sr=8000, x_axis='time', y_axis='mel',fmax=4000,cmap='Spectral_r')
    title("filtered expected mel spectrogram ")
    colorbar()
    
    plt.figure(figsize=(16, 6))
    plt.subplot(1,3,1)
    librosa.display.specshow(np.reshape(Spec_audio_decoded[display_index],(1024,16)), sr=8000, x_axis='time', y_axis='mel',fmax=4000,cmap='Spectral_r')
    title("mel spectrogram of the audio signal \n out of the audio network ")
    colorbar()
    
    plt.subplot(1,3,2)
    librosa.display.specshow(np.reshape(decoded_mel[display_index],(1024,16)), sr=8000, x_axis='time', y_axis='mel',fmax=4000,cmap='Spectral_r')
    title("output mel spectrogram of \n the mel network ")
    colorbar()
    
    plt.subplot(1,3,3)
   
    librosa.display.specshow(np.reshape(Spec_lin[display_index],(1024,16)), sr=8000, x_axis='time', y_axis='mel',fmax=4000,cmap='Spectral_r')
    title("output mel spectrogram of \n the lin_spec network ")
    colorbar()
    
    return

def display_audio_separately(display_index,Xtest,decoded_audio,merge_audio ,lim_sup=7999,lim_inf=0):
    xtest=np.reshape(Xtest[display_index],(8000))
    y=np.reshape(decoded_audio[display_index],(8000))
    
    plt.figure(figsize=(16, 4))
    plt.subplot(1,3,1)
    plt.plot(xtest[lim_inf:lim_sup],"r--")
    title("expected signal")
    plt.subplot(1,3,2)
    plt.plot(y[lim_inf:lim_sup],"b--")
    title(" audio signal out \n of the audio network")
    plt.subplot(1,3,3)
    plt.plot(merge_audio[display_index][lim_inf:lim_sup],"g--")
    title("audio signal after the merge \n of the mel spectrogram and the phase \n of the left plotted one")
    
    return

def display_audio_compare(display_index,Xtest,decoded_audio,merge_audio ,lim_sup=7999,lim_inf=0):
    xtest=np.reshape(Xtest[display_index],(8000))
    y=np.reshape(decoded_audio[display_index],(8000))
    plt.figure(figsize=(16, 4))
    plt.subplot(1,2,1)
    plt.plot(xtest[lim_inf:lim_sup]*1/np.linalg.norm(xtest),"r-",label="expected")
    plt.plot(y[lim_inf:lim_sup]*1/np.linalg.norm(y),"b-",label="output of audio network")
    title("expected signal vs audio network ouput")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(xtest[lim_inf:lim_sup]*1/np.linalg.norm(xtest),"r-",label="expected")
    plt.plot(merge_audio[display_index][lim_inf:lim_sup]*1/np.linalg.norm(merge_audio[display_index]),"g-",label="merge mel-audio")
    title("expected signal vs merge ouput")
    plt.legend()
    
    return
    
    
    