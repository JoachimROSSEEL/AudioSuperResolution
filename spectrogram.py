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


def dir_to_spectro(src,dst):
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
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(Zxx, ax=ax, y_axis='log', x_axis='time')
        fig.savefig(png_file)
    return
       
