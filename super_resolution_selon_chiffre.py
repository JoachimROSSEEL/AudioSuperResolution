#!/usr/bin/env python
# coding: utf-8

# In[135]:


import numpy
import scipy.signal
from matplotlib.pyplot import *
import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
from passe_bas_fonction import *
from spectrogram import *
#from free_spoken_digit_dataset.utils import spectogramer as sg #cf fsdd
import matplotlib.image as mpimg
import glob
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D, UpSampling2D, Input,Dropout,Conv2DTranspose,Concatenate


# In[2]:


original_dataset_dir = "C:\\Users\\joaro\\OneDrive\\Documents\\deep_learning_jupiter\\free_spoken_digit_dataset\\recordings\\"

base_dir = "C:\\Users\\joaro\\OneDrive\\Documents\\deep_learning_jupiter\\free_spoken_digit_dataset\\super_resolution"
train_dir = os.path.join(base_dir, 'train\\')

test_dir = os.path.join(base_dir, 'test\\')
train_dir_filtré=os.path.join(base_dir, 'train_dir_filtré\\')
test_dir_filtré=os.path.join(base_dir, 'test_dir_filtré\\')


# In[3]:


os.mkdir(base_dir)
os.mkdir(train_dir)
os.mkdir(test_dir)
os.mkdir(train_dir_filtré)
os.mkdir(test_dir_filtré)


# In[4]:


speakername=['_jackson_{}.wav','_nicolas_{}.wav','_theo_{}.wav','_yweweler_{}.wav']
for i in range(10):    
    for s in speakername:
        fnames_test=[str(i)+s.format(j) for j in range(5)]
        fnames_train=[str(i)+s.format(j) for j in range(5,50)]
        
        for fname in fnames_test:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(test_dir, fname)
            
            shutil.copyfile(src, dst)
            
        for fname in fnames_train:
             src = os.path.join(original_dataset_dir, fname)
             dst = os.path.join(train_dir, fname)
            
             shutil.copyfile(src, dst)
             
    i=i+1;


# In[3]:


#on doit creer un tableau d'image pour le test bruité, test non bruité , train bruité et train non bruité


dir_to_spectro(test_dir,test_dir,spectrogram_dimensions=(128, 128),  cmap='gray_r')


# In[5]:


print(train_dir)

dir_to_spectro(train_dir,train_dir,spectrogram_dimensions=(128, 128),  cmap='gray_r') 


# In[105]:


fce=0.09 #sr=22050 
ftype="butter"
ordre=4
b,a = creer_filter(fce,"highpass",ftype,ordre)
filter_path(train_dir,train_dir_filtré,b,a)
filter_path(test_dir,test_dir_filtré,b,a)


# In[106]:



dir_to_spectro(train_dir_filtré,train_dir_filtré,spectrogram_dimensions=(128, 128),  cmap='gray_r')

dir_to_spectro(test_dir_filtré,test_dir_filtré,spectrogram_dimensions=(128, 128),  cmap='gray_r')


# In[107]:


Xtrain_l=[]
Xtrain_filtered_l=[]
Xtest_l=[]
Xtest_filtered_l=[]

for filepath in sorted(glob.glob(os.path.join(test_dir, "*.png"))):
    Xtest_l.append(mpimg.imread(filepath))

print("ok")    
print(Xtest_l[0])
print(Xtest_l[0].shape)
imshow(Xtest_l[0])


# In[108]:


for filepath in sorted(glob.glob(os.path.join(train_dir, "*.png"))):
    Xtrain_l.append(mpimg.imread(filepath))    
for filepath in sorted(glob.glob(os.path.join(train_dir_filtré, "*.png"))):
    Xtrain_filtered_l.append(mpimg.imread(filepath))      
for filepath in sorted(glob.glob(os.path.join(test_dir_filtré, "*.png"))):
    Xtest_filtered_l.append(mpimg.imread(filepath))      


# In[109]:


Xtest = np.asarray(Xtest_l)    
Xtrain = np.asarray(Xtrain_l)  
print("milieu conv ok")
Xtest_filtered=np.asarray(Xtest_filtered_l)
Xtrain_filtered=np.asarray(Xtrain_filtered_l)
print("conv fini")

print(Xtrain.shape)
print(Xtest.shape)
print(Xtrain_filtered.shape)
print(Xtest_filtered.shape)


# In[110]:


plt.figure(figsize=(20, 4)) #affichage 10 premiers spectrogramme pour verifier que ajout est ok
n=10
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(Xtest[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display filtered
    ax = plt.subplot(2, n, i + 1+n)
    plt.imshow(Xtest_filtered[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
   
plt.show()


# In[111]:


print(Xtest[0].shape)


# In[178]:



#modele de https://arxiv.org/pdf/1703.08019.pdf
#pas séquentiel


input_img = Input(shape=(128, 128, 4))  # adapt this if using `channels_first` image data format

conv1= Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
max1= MaxPooling2D((2, 2), padding='same')(conv1)
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(max1)
encoded = MaxPooling2D((2, 2), padding='same')(conv2)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)



decoded = Conv2D(4, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.summary()

#2nd modele Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN) cf https://github.com/titu1994/Image-Super-Resolution 
#=> moins bien
# input_img = Input(shape=(128, 128, 4))  # adapt this if using `channels_first` image data format

# conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

# conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
# encoded = Conv2DTranspose(32,(3, 3), padding='same')(conv2)

# merge1=Concatenate(axis=-1)([encoded,x2])
# deconv1=Conv2DTranspose(32,(3, 3), padding='same')(merge1)
# merge2=Concatenate(axis=-1)([deconv1,x1])

# decoded = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)


# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='mse')

# autoencoder.summary()


# In[179]:


autoencoder.fit(Xtrain_filtered,Xtrain,epochs=60,
                batch_size=20,
                shuffle=True,
                validation_data=(Xtest_filtered, Xtest))


# In[180]:


decoded_imgs = autoencoder.predict(Xtest_filtered)


# In[182]:


n = 5
plt.figure(figsize=(40, 40))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(Xtest[i])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #display original filtered
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(Xtest_filtered[i])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n+1+n)
    plt.imshow(decoded_imgs[i])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[183]:


import IPython.display as ipd
import soundfile
sr=22050
t,y=scipy.signal.istft(abs(decoded_imgs[0]))
print(y)
audio_path_record="C:\\Users\\joaro\\OneDrive\\Documents\\deep_learning_jupiter\\free_spoken_digit_dataset\\"
audio_path_record=os.path.join(audio_path_record,"0_jackson_0_reconstructed.wav")
soundfile.write(audio_path_record,y,sr) #soundwrite mieux que la fonction dans librosa
ipd.Audio(audio_path_record)


# In[ ]:




