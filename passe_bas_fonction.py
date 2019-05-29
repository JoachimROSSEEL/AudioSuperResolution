#!/usr/bin/env python
# coding: utf-8




#calcul du filtre avec fce=fréquence de coupure numérique et btype= type de filtre 
#ftype= nom du filtre(Butterworth...) et  ordre= ordre du filtre
#rs et rp sont pour les filtres de Chebychev et Elliptique
#renvoie le filtre
import scipy.signal
import os, shutil
import librosa
import numpy
import IPython.display as ipd
import glob
import soundfile
import IPython.display as ipd

def creer_filter(fce,btype,ftype,ordre,*rs, **rp):    
    b,a = scipy.signal.iirfilter(N=ordre,Wn=fce*2,btype=btype,ftype=ftype)
    return b,a

#x tableau 1D= signal audio temporel et renvoie le signal filtré    
def filtrage(b,a,x):
    
    return scipy.signal.lfilter(b, a, x)





#filtre tout les fichiers wav d'un dossier de chemin src et le stocke dans dst sous forme de .wav
#voir fonction avec spectro de audiomnist pour s'inspirer
#b,a sont les paramètres du filtre numérique
#inspirer de preprocess_data de audioMNIST 
def filter_path(src,dst,b,a):
        
    if not os.path.exists(dst):
          os.makedirs(dst)
         
    print("processing {}".format(src))
  
    
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
        #on va filtrer
        x , sr = librosa.load(filepath)
        
        y=filtrage(b,a,x)
        
        # infer sample info from name
        dig, nom, essai = filepath.rstrip(".wav").split("\\")[-1].split("_")
        #on va filtrer
     
        
        wav_file=os.path.join(dst,'filtered_{}_{}_{}.wav'.format(dig,nom,essai))
        soundfile.write(wav_file,y,sr)
        
                                    
    return
   
                    







