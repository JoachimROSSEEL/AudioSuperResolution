# -*- coding: utf-8 -*-
#reprise du code de preprocessing data de audioMnist

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
       
