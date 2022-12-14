import os
import pandas as pd
import librosa as lr
import librosa.display 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import python_speech_features as mfcc
import joblib
from sklearn import preprocessing
from librosa import power_to_db , util
import scipy
iter = 2


def calculate_delta(array):
    rows, cols = array.shape
    # print(rows)
    # print(cols)
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas

def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=2205, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    #     print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined
def plot_melspectrogram(file_name):
    plt.rcParams['font.size'] = '20'
    audio,sfreq = lr.load(file_name)
    melspectrogram = lr.feature.melspectrogram(y=audio, sr=sfreq)
    fig=plt.figure(figsize=(25,10))
    img=librosa.display.specshow(melspectrogram,x_axis='time',y_axis='mel',sr=sfreq)
    fig.colorbar(img,format="%+2.f")
    plt.savefig('./static/spectro.png')
    return img,fig
def apply_model(features_list):
    # voice_model = pickle.load(open('./Person_model.sav', 'rb'))
    # speech_model=pickle.load(open('./Word_model.sav', 'rb'))
    # x_pre = np.array(features_list)
    audio, sr = lr.load(features_list, sr=48000, mono=True,duration=30)
    vector = extract_features(audio,sr)
    gmm_files = [i + '.joblib' for i in ['Misara', 'Ahmed', 'Youssef','Hanya']]
    models = [joblib.load(fname) for fname in gmm_files]
    log_likelihood = np.zeros(len(models))
    y=[]
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    flag = False
    flagLst = log_likelihood-max(log_likelihood)
    winner = np.argmax(log_likelihood)
    for i in range(len(flagLst)):
        if flagLst[i] == 0:
            continue
        if abs(flagLst[i])<0.4:
            flag = True
    if flag:
        winner = 4
    y.append(log_likelihood)
    gmm_files = [i + '.joblib' for i in ['Door', 'Close', 'Book','Window']]
    models = [joblib.load(fname) for fname in gmm_files]
    log_likelihood = np.zeros(len(models))
    y=[]
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    winner_0 = np.argmax(log_likelihood)
    # x_pre=x_pre.reshape(1,-1)
    # # voice_prediction=voice_model.predict(x_pre)
    # # speech_prediction=speech_model.predict(x_pre)
    pred_num = []
    pred_num.append([winner_0])
    pred_num.append([winner])
    # pred_num.append(voice_prediction)
    # pred_num.append(speech_prediction)
    pred_num = np.array(pred_num)
    print(pred_num)
    return pred_num
def Names_return(a):
    """return list of names according to list of numbers 
    first item 0-> Open
    1-> Not def
    2-> Close
    second item 127.0.0.1 - - [13/Dec/2022 15:44:18] "GET / HTTP/1.1" 500 -

    0-> Ahmed
    1-> Hanya
    2-> Misara
    3-> Youssef
    4-> Others"""
    voice = ['Door', 'Close', 'Book','Window']
    speech = ["Misara","Ahmed","Youssef","Hanya","Others"]
    print(a)
    names = []
    names.append(voice[a[0][0]])
    names.append(speech[a[1][0]])
    return names