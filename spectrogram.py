# -*- coding: utf-8 -*-
#reprise d'une partie code de preprocessing data de audioMnist
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
#the arrays contained in Spec_l will be of the following format: [Spec_real,Spec_imag] with Spec_real of shape (1025,12) for the moment
def dir_to_spectro_RI(src):
    Spec_l=[]
#     Spec_real_l=[]
#     Spec_imag_l=[]
    zero_pad=8000
    print("processing {}".format(src))
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
        data, fs =librosa.load(filepath)
        # resample , shannon still okay
        data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
        #zero_padding, in order to avoid bug and to have the same shape for all data in lenght time
        if len(data) > zero_pad: #ok si durée d'une seconde
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
#         Spec_real_l.append(np.real(Spec)/(np.real(Spec).max()))
#         Spec_imag_l.append(np.imag(Spec)/(np.imag(Spec).max()))
        Ntab=np.reshape([np.real(Spec)/(np.real(Spec).max()),np.imag(Spec)/(np.imag(Spec).max())],(1025,16,2)) #on normalise les valeurs pour que tout soit entre 0 et 1
        
        Spec_l.append(Ntab)
        
        
#     Spec_real=np.asarray(Spec_real_l)
#     Spec_imag=np.asarray(Spec_imag_l)
    Spec_t=np.asarray(Spec_l)
    return Spec_t
#create png file to represent the complex in the C space       

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    