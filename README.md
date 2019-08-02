# AudioSuperResolution
It's a project which aims to reconstruct harmonics from a filtered signal with an autoencoder. For this branch, I am working on super resolution with spoken digit from FSDD. 

All the .py contain functions related to their names. They are used along the code, such during the preprocessing for example.

The file super_resolution_selon_chiffre.ipynb was a test to do classification according to the pronounced number.

The file super_resolution_number_with_png.ipynb does a super resolution with spectrograms stored as png files. 

The working autoencoders are the following:
        audio network, with the file super_resolution_with_wav.ipynb
        mel network, with the file super_resolution_melspectro.ipynb
        mel + audio networks, with the file super_resolution_with_wav_and_mel.ipynb
        lin + audio networks, with the file super_resolution_with_wav_and_lin.ipynb

You can only have a look at the code as you will lack the data to run those notebooks.
        
However, the file test_notebook.ipynb is the main file in order to test the audio network, the mel network, the lin network and the merge method. It allows to make some tests of your own.


You can read the report with project_report.pdf.
