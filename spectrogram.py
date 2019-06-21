# -*- coding: utf-8 -*-
#part of the code of preprocessing data de audioMnist
#those functions allows to create and to do some manipulation with spectrograms
import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wavf
import scipy.signal
import librosa
import multiprocessing
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#this functions stores the magnitude spectrogram as a png file. They have a log frequency scale and a log ampltude
def dir_to_spectro(src,dst, spectrogram_dimensions=(64, 64),  cmap='gray_r'):
    # create folder for png files
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
      
        # infer sample info from name
        dig, nom, essai = filepath.rstrip(".wav").split("\\")[-1].split("_")
        # read data
        data, fs = librosa.load(filepath)
        
        # resample
        

        Zxx = librosa.stft(data)
        
        # convert to decibel
        Zxx = librosa.amplitude_to_db(abs(Zxx))
        #save as png
        png_file=os.path.join(dst,'{}_{}_{}.png'.format(dig,nom,essai))
        #imgpil = Image.fromarray(Zxx) 
        #imgpil.convert('RGB').save(png_file,"PNG")
      
        fig = plt.Figure()
        fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
        canvas = FigureCanvas(fig)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        p = librosa.display.specshow(Zxx, ax=ax, y_axis='log', x_axis='time',cmap=cmap)
        fig.savefig(png_file)
    return



#this functions calculates stores all the real and imaginary part of a folder containing wav file in two separates arrays which will later be feed to the autoencoder. 
#the arrays contained in Spec_l will be of the following format:  (1024,16,2) for the moment, and lr_max, li_max containes the max uses for normalization of the spectrograms so you can remultpiply by it after the learning to restore the correct amplitude

def dir_to_spectro_RI(src):
    Spec_l=[]
    lr_max=[]
    li_max=[]

    zero_pad=8000
    print("processing {}".format(src))
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
        data, fs =librosa.load(filepath)
        # resample , shannon still okay
        data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
        #zero_padding, in order to avoid bug and to have the same shape for all data in lenght time
        if len(data) > zero_pad: #our data here last 1 second maximun 
            print(filepath)
            
            embedded_data=data[0:zero_pad]
        elif len(data) < zero_pad:
            embedded_data = np.zeros(zero_pad)
            
            embedded_data[0:len(data)] = data
            
        elif len(data) == zero_pad:
            # nothing to do here
            embedded_data = data
            pass

        Spec= librosa.stft(embedded_data, n_fft=2048,window='hann')
        
        lr_max.append(np.real(Spec).max())
        li_max.append(np.imag(Spec).max())
        Specr=np.real(Spec)/(np.real(Spec).max())
        Speci=np.imag(Spec)/(np.imag(Spec).max())
        
        image_ri = np.zeros((1024, 16, 2)) #we remove the higher frequency to have 1024 as a dimension so a power of two , resolution of 4Hz between two indices
        for i in range(1024):
            for j in range(16):
                image_ri[i][j][0]=Specr[i][j]
                image_ri[i][j][1]=Speci[i][j]
                #as a pixel containing the real and imaginary part, same way of a color
                j=j+1   
            i=i+1    
                   

        
        Spec_l.append(image_ri)
        
      
    Spec_t=np.asarray(Spec_l)
    return Spec_t,lr_max,li_max   

#this functions stores all the wav_file of a src folder into a array containing their mel spectrogram
#It also calculate and return their phase for the moment
#Why mel? mel is similar to a log scale ,so it assures that the low frequency information is as represented as the high frequency
def dir_to_log_mel(src):
    Spec_l=[]
    Phase_l=[]
    zero_pad=8000
    print("processing {}".format(src))
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
        data, fs =librosa.load(filepath)
        # resample , shannon still okay
        data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
        #zero_padding, in order to avoid bug and to have the same shape for all data in lenght time
        if len(data) > zero_pad: #our data here last 1 second maximun 
            print(filepath)
            
            embedded_data=data[0:zero_pad]
        elif len(data) < zero_pad:
            embedded_data = np.zeros(zero_pad)
            
            embedded_data[0:len(data)] = data
            
        elif len(data) == zero_pad:
            # nothing to do here
            embedded_data = data
            pass
        Stft=librosa.stft(embedded_data, n_fft=2048,window='hann')
        Spec= librosa.feature.melspectrogram(S=Stft, sr=8000, n_fft=2048, hop_length=512, power=2.0,n_mels=1024,norm=None)
        Mag,Phase=librosa.core.magphase(Stft, power=1)
        Phase_l.append(Phase[0:Phase.shape[0]-1,:])
       
        Spec=np.log(abs(Spec)+1)
        #log amp for better learning of the neural network
        #+1 to avoid log(0.0)
        
        Spec_l.append(np.reshape(Spec,(Spec.shape[0],Spec.shape[1],1))) 
        
    return np.asarray(Spec_l),np.asarray(Phase_l) 
        

# inspired by the of code at https://github.com/4p0pt0Z/Audio_blind_source_separation/blob/master/data_set.py
#mel_filterbank => cf https://librosa.github.io/librosa/generated/librosa.filters.mel.html
#fct à utiliser: librosa.filters.mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1)         
#Mel_t is an array which  le tableau having for element i a spectrogram of shape (1024,16,1) 
#rest of arguments are those of the fct librosa.filters.mel

def mel_to_linspec(Mel_t,sr,n_mels,n_fft,fmin=0.0,fmax=None):
    Spec_t=np.zeros((Mel_t.shape[0],Mel_t.shape[1],Mel_t.shape[2]),dtype=complex)
    mel_filter=librosa.filters.mel(sr,n_fft,n_mels,fmin,fmax,norm=None) #matrix of passage to go from linear spec to mel spec
    inverse_mel_filter= np.linalg.pinv(mel_filter) #we reverse the matrix of passage to go from mel_spec to linear spec
    
    for i in range(len(Mel_t)):
        Mel_t[i] = np.power(10.0 * np.ones(Mel_t[i].shape), (Mel_t[i] / 10.0)) - 1.0 #-1 because of the way it's calculates in dir_to_log_mel
        
        Spec_t[i]=inverse_mel_filter[0:inverse_mel_filter.shape[0]-1,:] @ np.reshape(Mel_t[i],(Mel_t[i].shape[0],Mel_t[i].shape[1]))
        
    return Spec_t
        
        
        
        
        
        
        
        
    