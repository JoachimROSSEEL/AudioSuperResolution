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

def dir_to_spectro(src,dst,spectrogram_dimensions=(64, 64), cmap='gray_r'):
    # create folder for png files
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
      
        # infer sample info from name
        dig, nom, essai = filepath.rstrip(".wav").split("\\")[-1].split("_")
        # read data
        fs, data = wavf.read(filepath)
        
        # resample
        data = librosa.core.resample(y=data.astype(np.float32), orig_sr=fs, target_sr=8000, res_type="scipy")
        # zero padding
        if len(data) > 8000:
            raise ValueError("data length cannot exceed padding length.")
        elif len(data) < 8000:
            embedded_data = np.zeros(8000)
            offset = np.random.randint(low = 0, high = 8000 - len(data))
            embedded_data[offset:offset+len(data)] = data
            
        elif len(data) == 8000:
            # nothing to do here
            embedded_data = data
            pass
        Zxx = librosa.stft(embedded_data)
        
        # convert to decibel
        Zxx = librosa.amplitude_to_db(abs(Zxx))
        #save as png
        png_file=os.path.join(dst,'{}_{}_{}.png'.format(dig,nom,essai))
        
      
        
        return
       
