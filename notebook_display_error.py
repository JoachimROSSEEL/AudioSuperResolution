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

from error_measure import *

#This file contains functions used in the notebook in order to display error measurements. 
def display_energy_spectrum(Xtest):
    spectrum_speech_audio_mnist = np.abs(np.fft.rfft(np.reshape(Xtest,(Xtest.shape[0],Xtest.shape[1],Xtest.shape[2])))).sum(axis=2).sum(axis=0)
    spectrum_speech_audio_mnist_n = spectrum_speech_audio_mnist/np.max(spectrum_speech_audio_mnist)*0.1/len(Xtest)
    fs=np.zeros(1024)
    for i in range(1024):
        fs[i]=4*i
        i=i+1

    fig=plt.figure(figsize=(16, 4))
    plt.xlabel("frequencies(Hz) ")
    plt.ylabel('energy magnitude')
    plt.fill_between(fs, spectrum_speech_audio_mnist_n, alpha = 5) #alpha set the brightness 
    plt.title('repartition of the normalized energy for our dataset')
    return fs,spectrum_speech_audio_mnist_n

def display_mel_mse_highpass(fs,energy_spectrum,alpha=0.2):
    fch=[100,250,400,500,650,800,900,1000,1500,2000,3000]
    mel_mse_expected_fil=[4.899083107907589e-07,6.581620565635553e-06 ,1.4253546064176374e-05 ,2.02798452596404e-05,3.059972105987759e-05
,4.1077088562028885e-05 ,4.7306026989506134e-05 ,5.2756993255817805e-05 ,6.900860649714932e-05 ,7.839318563822392e-05 ,9.073598754380241e-05]

    mel_mse_expected_audio=[8.45009023533109e-06 ,9.045923478437584e-06 ,1.0539854936830291e-05 ,1.073740133854983e-05 ,1.2050821072378661e-05 ,1.1761112717721592e-05 ,1.1896799871816977e-05 ,1.3348344849190107e-05 ,1.361908796958397e-05 ,1.6216031844259575e-05,2.075509121916088e-05 ]

    mel_mse_expected_mel=[6.334015694622778e-07,1.93701487728076e-06 ,4.221574206456429e-06 ,6.894075064905749e-06 ,9.015606819444774e-06 ,1.1789361457691057e-05,1.3618950343630778e-05 ,1.8214523466708357e-05 ,2.584721615496709e-05 ,3.023515332637395e-05 ,4.024768349553545e-05]

    mel_mse_expected_lin=[4.647551425140125e-07 ,1.2406480203389031e-06 ,3.7208390484252993e-06 ,4.485565901189787e-06 ,6.75512599522486e-06 ,9.585675407773428e-06 ,1.0163672566998292e-05 ,1.2298150215414614e-05 ,2.1350239011158213e-05 ,2.3143372866831834e-05 ,3.296722636497962e-05 ]

    fig=plt.figure(figsize=(16, 4))

    plt.plot(fch,mel_mse_expected_fil,"r",label="mel_mse_expected_fil")
    plt.plot(fch,mel_mse_expected_audio,"b",label="mel_mse_expected_audio")
    plt.plot(fch,mel_mse_expected_mel,"g",label="mel_mse_expected_mel")
    plt.plot(fch,mel_mse_expected_lin,"orange",label="mel_mse_expected_lin")
    plt.fill_between(fs, energy_spectrum*0.2, alpha=alpha)
    plt.xlabel("frequencies(Hz) ")
    plt.ylabel('mse')
    title("evolution of differents \n mel spectrogram mse according \n to the cut off frequency(highpass)" )
    plt.legend()
    plt.show()
    
    return

def display_audio_mse_highpass(fs,energy_spectrum,alpha=0.2):
    fch=[100,250,400,500,650,800,900,1000,1500,2000,3000]
    mse_expected_fil=[1.446375029707019e-06 ,1.5678172574829692e-05 ,3.422177307968991e-05 ,5.0073371002029336e-05,7.760537658108146e-05 
,0.0001032307532074666 ,0.00011728559086309962 ,0.00012890817369316522 ,0.00016148241079721865,0.00018060181913864232,0.00020083367439990396 ]

    mse_expected_audio=[1.1709043717600109e-06,1.5464223496520057e-06 ,7.452432831659676e-06 ,8.474299870313555e-06,1.0653167908426954e-05 
,1.1689741487763436e-05 ,1.188702350233708e-05 ,1.494346405408763e-05 ,1.8208046916599074e-05 ,2.822905268333845e-05 ,5.0331236627590626e-05]

    mse_expected_mel=[4.195195264581628e-06,6.807078485635804e-06 ,1.4196958351330743e-05,2.3719182163922992e-05,2.5377911629047107e-05 ,3.490540757989077e-05 ,3.789871863143999e-05 ,3.850738800504175e-05 ,6.398863878809281e-05 ,8.124327950242894e-05 ,0.00010909449785205259]

    fig=plt.figure(figsize=(16, 4))

    plt.plot(fch,mse_expected_fil,"r",label="mse_expected_fil")
    plt.plot(fch,mse_expected_audio,"b",label="mse_expected_audio")
    plt.plot(fch,mse_expected_mel,"g",label="mse_expected_merge")
    plt.fill_between(fs, energy_spectrum*0.5, alpha = alpha)
    plt.xlabel("frequencies(Hz)")
    plt.ylabel('mse')
    title("evolution of differents \n audio mse according \n to the cut off frequency(highpass)" )
    plt.legend()
    plt.show()
    
    return

def display_mel_mse_lowpass(fs,energy_spectrum,alpha=0.2):
    fcl=[4000,3000,2000,1500,1250,1000,900,800,650,500,400,250]
    mel_mse_expected_fil_low=[2.970063307496591e-08 ,1.7565723465168984e-07 ,8.57166646326088e-07 ,1.8932897030705133e-06 ,2.8500279455035766e-06 ,4.370139798289199e-06 
,5.248268424871246e-06 ,6.384213925059141e-06,8.971952127489672e-06 ,1.3715875230204185e-05 ,1.959227138747852e-05 ,3.8437237737175776e-05]

    mel_mse_expected_audio_low=[8.241889547638393e-06 ,8.250499822085686e-06 ,8.062604316956318e-06 ,8.232580351346358e-06 ,8.857037761472599e-06 
,8.535990040496815e-06 ,8.461746104001673e-06 ,8.809810810809413e-06 ,9.215930015779404e-06 ,9.666199877775872e-06 ,1.0162382440525922e-05 ,2.165543602286896e-05 ]

    mel_mse_expected_mel_low=[6.504844575351824e-07 ,6.474844575351824e-07 ,6.33225773649575e-07 ,9.688656962073065e-07 ,1.2165758144166473e-06 ,1.821634106238542e-06 
,2.596936924653463e-06 ,2.62623405307173e-06 ,4.403196902397579e-06 ,6.32003388982315e-06 ,9.141212265247093e-06 ,1.0850531151454406e-05 ]

    mel_mse_expected_spec_lin=[1.5339491874456822e-07 ,3.1205061773227365e-07 ,1.3452088188558786e-06 ,1.5728389255738134e-06 ,2.3589043835506954e-06 ,2.4690590210029707e-06 
,3.3453113774510083e-06 ,3.4638246599479217e-06 ,4.016796046266995e-06 ,4.996016976333851e-06,6.046828578960423e-06 ,9.208343087152952e-06 ]

    fig=plt.figure(figsize=(16, 4))

    plt.plot(fcl,mel_mse_expected_fil_low,"r",label="mel_mse_expected_fil")
    plt.plot(fcl,mel_mse_expected_audio_low,"b",label="mel_mse_expected_audio")
    plt.plot(fcl,mel_mse_expected_mel_low,"g",label="mel_mse_expected_mel")
    plt.plot(fcl,mel_mse_expected_spec_lin,"orange",label="mel_mse_expected_spec_lin")
    plt.fill_between(fs, energy_spectrum*0.1, alpha = alpha)
    plt.xlabel("frequencies(Hz) ")
    plt.ylabel('mse')
    title("evolution of differents \n mel spectrogram mse according \n to the cut off frequency(low pass)" )
    plt.legend()
    plt.show()
    return

def display_audio_mse_lowpass(fs,energy_spectrum,alpha=0.2):
    fcl=[4000,3000,2000,1500,1250,1000,900,800,650,500,400,250]
    mse_expected_fil_low=[1.6203385335490284e-07 ,8.165222338088468e-07 ,2.8091898268963365e-06 ,4.852340894495716e-06 ,6.481584850449539e-06 ,9.047599874756072e-06 ,1.0595423889640227e-05 ,1.2702025337879221e-05 ,1.8012244441410045e-05 ,2.931733166743327e-05 ,4.4549469602745724e-05 ,9.15950801589359e-05 ]

    mse_expected_audio_low=[1.57767767832426e-06 ,1.0968487705916454e-06 ,3.0711164416932655e-06 ,4.516285423029928e-06 ,5.1625036600363515e-06 ,5.4903515360170494e-06 ,5.686809621408645e-06 ,6.138657538535576e-06 ,6.955787688031216e-06 ,8.497511000634544e-06 ,1.0699306705760529e-05 ,9.50140637215458e-05]

    mse_expected_mel_low=[3.626645623102678e-06 ,3.898683608250555e-06 ,4.245712831327584e-06 ,5.321634475830765e-06 ,5.727008790592791e-06 ,6.493210180226599e-06 ,7.495743276972614e-06 ,8.30521211312121e-06 ,1.1420552850136243e-05 ,1.7521317990358807e-05 ,2.3950751688499205e-05 ,5.214530415874166e-05 ]

    mel_mse_expected_spec_lin=[1.5339491874456822e-07 ,3.1205061773227365e-07 ,1.3452088188558786e-06 ,1.5728389255738134e-06 ,2.3589043835506954e-06 ,2.4690590210029707e-06 
,3.3453113774510083e-06 ,3.4638246599479217e-06 ,4.016796046266995e-06 ,4.996016976333851e-06,6.046828578960423e-06 ,9.208343087152952e-06 ]
    fig=plt.figure(figsize=(16, 4))

    plt.plot(fcl,mse_expected_fil_low,"r",label="mse_expected_fil")
    plt.plot(fcl,mse_expected_audio_low,"b",label="mse_expected_audio")
    plt.plot(fcl,mse_expected_mel_low,"g",label="mse_expected_merge")

    plt.fill_between(fs, energy_spectrum*0.2, alpha = alpha)
    plt.xlabel("frequencies(Hz) ")
    plt.ylabel('mse')
    title("evolution of differents \n audio mse according \n to the cut off frequency(lowpass)" )
    plt.legend()
    plt.show()
    
    return

def display_log_mel_mse_lowpass():
    fcl=[4000,3000,2000,1500,1250,1000,900,800,650,500,400,250]
    mel_mse_expected_fil_low=[2.970063307496591e-08 ,1.7565723465168984e-07 ,8.57166646326088e-07 ,1.8932897030705133e-06 ,2.8500279455035766e-06 ,4.370139798289199e-06 
,5.248268424871246e-06 ,6.384213925059141e-06,8.971952127489672e-06 ,1.3715875230204185e-05 ,1.959227138747852e-05 ,3.8437237737175776e-05]

    mel_mse_expected_mel_low=[6.504844575351824e-07 ,6.474844575351824e-07 ,6.33225773649575e-07 ,9.688656962073065e-07 ,1.2165758144166473e-06 ,1.821634106238542e-06 
,2.596936924653463e-06 ,2.62623405307173e-06 ,4.403196902397579e-06 ,6.32003388982315e-06 ,9.141212265247093e-06 ,1.0850531151454406e-05 ]

    mel_mse_expected_spec_lin=[1.5339491874456822e-07 ,3.1205061773227365e-07 ,1.3452088188558786e-06 ,1.5728389255738134e-06 ,2.3589043835506954e-06 ,2.4690590210029707e-06 
,3.3453113774510083e-06 ,3.4638246599479217e-06 ,4.016796046266995e-06 ,4.996016976333851e-06,6.046828578960423e-06 ,9.208343087152952e-06 ]
    fig=plt.figure(figsize=(16, 4))
    plt.plot(fcl,np.log(mel_mse_expected_fil_low),"r",label="mse_expected_fil")
    plt.plot(fcl,np.log(mel_mse_expected_spec_lin),"g",label="mse_expected_spec_lin")
    plt.plot(fcl,np.log(mel_mse_expected_mel_low),"orange",label="mse_expected_mel")

    plt.xlabel("frequencies(Hz) ")
    plt.ylabel('mse')
    title("evolution of differents \n audio mse according \n to the cut off frequency(lowpass)" )
    plt.legend()
    plt.show()
    
    return
