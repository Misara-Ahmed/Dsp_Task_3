import os
import pandas as pd
import librosa as lr
import librosa.display 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from librosa import power_to_db , util
import scipy




def plot_melspectrogram(file_name):
    
    plt.rcParams['font.size'] = '20'
    audio,sfreq = lr.load(file_name)
    melspectrogram = lr.feature.melspectrogram(y=audio, sr=sfreq)
    fig=plt.figure(figsize=(25,10))
    img=librosa.display.specshow(melspectrogram,x_axis='time',y_axis='mel',sr=sfreq)
    fig.colorbar(img,format="%+2.f")
    plt.savefig('./static/spectro.png')
    return

def apply_model(features_list):
    voice_model = pickle.load(open('./Person_model.sav', 'rb'))
    speech_model=pickle.load(open('./Word_model.sav', 'rb'))
    x_pre=np.array(features_list)
    x_pre=x_pre.reshape(1,-1)
    voice_prediction=voice_model.predict(x_pre)
    speech_prediction=speech_model.predict(x_pre)
    return voice_prediction,speech_prediction
