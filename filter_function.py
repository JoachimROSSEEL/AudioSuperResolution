#!/usr/bin/env python
# coding: utf-8




#filter calculation with fce = normalized frequency of filter
# ftype = filter name (Butterworth ...) and order = filter order
#rs and rp are for Chebychev and Elliptical filters

import scipy.signal
import os, shutil
import librosa
import numpy
import IPython.display as ipd
import glob
import soundfile
import IPython.display as ipd
#create and return the coefficients of the filters
def creer_filter(fce,btype,ftype,ordre,*rs, **rp):    
    b,a = scipy.signal.iirfilter(N=ordre,Wn=fce*2,btype=btype,ftype=ftype)
    return b,a

#x is a 1D array, generally the wav file converted into an array. This functions return the filtered signal.    
def filtrage(b,a,x):
    
    return scipy.signal.lfilter(b, a, x)






#filter all the wav files from a src path folder and store it in dst as .wav

# b, a are the parameters of the digital filter
#Inspire from preprocess_data of audioMNIST
def filter_path(src,dst,b,a):
        
    if not os.path.exists(dst):
          os.makedirs(dst)
         
    print("processing {}".format(src))
  
    
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
        #load the wav file
        x , sr = librosa.load(filepath)
        
        #filtering
        y=filtrage(b,a,x)
        
        # infer sample info from name
        dig, nom, essai = filepath.rstrip(".wav").split("\\")[-1].split("_")
       
     
        #storing
        wav_file=os.path.join(dst,'{}_{}_{}.wav'.format(dig,nom,essai))
        soundfile.write(wav_file,y,sr)
        
                                    
    return
   
                    







