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
        normalized_data=embedded_data 
        wav_l.append(np.reshape(embedded_data,(zero_pad,1,1)))
          
    return np.asarray(wav_l),np.asarray(max_l)