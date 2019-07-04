import numpy as np
import glob
import os
import librosa

#put all the wav files of a folder into an array
#return this array and an array containing each max of the wav file
def dir_to_wav_array(src):
    wav_l=[]
    max_l=[]

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
        
        max_l.append(embedded_data.max())
        normalized_data=embedded_data/embedded_data.max() 
        wav_l.append(np.reshape(normalized_data,(zero_pad,1,1)))
          
    return np.asarray(wav_l),np.asarray(max_l)

#this functions calculates an audio signal by taking Spec_t[i], a spectrgram of shape (1024,16) and multiplying it by the phase of the stft of audio[i]
#Then,after an istft it returns an array of audio data with a merge phase and magnitude
def merge_specphase_to_audio(Spec_t,audio):
    lenght=len(Spec_t)
    merge_audio=np.zeros((audio.shape[0],audio.shape[1]))
    
    if(lenght!=len(audio)):
        raise ValueError("the two arrays must have the same lenght")
    for i in range(lenght):
        Audio_stft=librosa.stft(np.reshape(audio[i],(audio.shape[1])), n_fft=2048,window='hann')
        Mag,Phase=librosa.magphase(Audio_stft,2)
        Mix_stft=Spec_t[i]*Phase[0:1024,:] 
        Audio=np.concatenate((Mix_stft,np.zeros((1,16),dtype=complex)),axis=0)  #to have shape 1025,16
        data=np.asarray(librosa.istft(Audio,length=8000))
        merge_audio[i]=data
        i=i+1
    return merge_audio
        
        
    
    
    
    
    
    
    
    
    