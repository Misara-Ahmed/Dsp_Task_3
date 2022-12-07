import os
import pandas as pd
import librosa as lr
import librosa.display 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from librosa import power_to_db , util
import scipy
import csv
# def extract_features(file_name):
#     audio,sfreq = lr.load(file_name)
#     S = np.abs(lr.stft(audio))
#     pitches,magnitudes = lr.core.piptrack(y = audio ,sr = sfreq)
#     #min_pitch = np.min(pitches)
#     max_pitch = np.max(pitches)
#     avg_pitch = np.mean(pitches)
#     var_pitch = np.var(pitches)
#     y_harmonic, y_percussive = lr.effects.hpss(audio)
#     #print(np.mean(y_harmonic),np.mean(y_percussive))
#     harmonic = np.mean(y_harmonic)
#     harmonic_var = np.var(y_harmonic)
#     percussive = np.mean(y_percussive)
#     percussive_var = np.var(y_percussive)
#     chroma=lr.feature.chroma_cens(y=y_harmonic, sr=sfreq)
#     #print(np.mean(chroma))
#     Chroma_cens = np.mean(chroma)
#     Chroma_cens_var = np.var(chroma)
    
#     chroma_stft =lr.feature.chroma_stft(y=y_harmonic, sr=sfreq)
#     chroma_stft_mean = np.mean(chroma_stft)
#     chroma_stft_var = np.var(chroma_stft)
    
#     chroma_cqt =lr.feature.chroma_cqt(y=y_harmonic, sr=sfreq)
#     chroma_cqt_mean = np.mean(chroma_cqt)
#     chroma_cqt_var = np.var(chroma_cqt)
    
#     mfccs = lr.feature.mfcc(y=y_harmonic, sr=sfreq)
#     #print(np.mean(mfccs))
#     Mfccs = np.mean(mfccs)
#     Mfccs_var = np.var(mfccs)
#     delta = lr.feature.delta(mfccs)
#     mfcc_delta_mean = np.mean(delta)
#     mfcc_delta_var = np.var(delta)
#     contrast=lr.feature.spectral_contrast(y=y_harmonic,sr=sfreq)
#     #print(np.mean(contrast))
#     Contrast = np.mean(contrast)
#     Contrast_var = np.var(contrast)
    
#     rolloff = lr.feature.spectral_rolloff(y=audio, sr=sfreq)
#     #print(np.mean(rolloff))
#     Rolloff = np.mean(rolloff)
#     Rolloff_var = np.var(rolloff)
    
#     zrate=lr.feature.zero_crossing_rate(y_harmonic)
#     #print(np.mean(zrate) )
#     Zrate = np.mean(zrate)
#     Zrate_var = np.var(zrate)
    
#     cent = lr.feature.spectral_centroid(y=audio, sr=sfreq)
#     Cent = np.mean(cent)
#     Cent_var = np.var(cent)
    
#     tonnetz = lr.feature.tonnetz(y=audio, sr=sfreq)
#     tonnetz_mean = np.mean(tonnetz)
#     tonnetz_var = np.var(tonnetz)
    
#     poly_features = lr.feature.poly_features(S=S, sr=sfreq)
#     poly_features_mean = np.mean(poly_features)
#     poly_features_var = np.var(poly_features)
    
#     spec_bw = lr.feature.spectral_bandwidth(y=audio, sr=sfreq)
#     spec_bw_mean = np.mean(spec_bw)
#     spec_bw_var = np.var(spec_bw)
    
#     rmse = lr.feature.rms(y=audio)
#     rmse_mean = np.mean(rmse)
#     rmse_var = np.var(rmse)
    
#     melspectrogram = lr.feature.melspectrogram(y=audio, sr=sfreq)
#     melspec_mean = np.mean(melspectrogram)
#     melspec_var = np.var(melspectrogram)
    
#     data = list([max_pitch,avg_pitch,var_pitch,harmonic,harmonic_var,percussive,percussive_var,Chroma_cens,Chroma_cens_var,chroma_stft_mean,chroma_stft_var,chroma_cqt_mean,chroma_cqt_var,Mfccs,Mfccs_var,mfcc_delta_mean,mfcc_delta_var,Contrast,Contrast_var,Rolloff,Rolloff_var,Zrate,Zrate_var,Cent,Cent_var,tonnetz_mean,tonnetz_var,poly_features_mean,poly_features_var,spec_bw_mean,spec_bw_var,rmse_mean,rmse_var,melspec_mean,melspec_var])
#     return data

def rms(y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode="constant",):
    if y is not None:
        if center:
            padding = [(0, 0) for _ in range(y.ndim)]
            padding[-1] = (int(frame_length // 2), int(frame_length // 2))
            y = np.pad(y, padding, mode=pad_mode)

        x = util.frame(y, frame_length=frame_length, hop_length=hop_length)

        # Calculate power
        power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    elif S is not None:
        # Check the frame length
        if S.shape[-2] != frame_length // 2 + 1:
            raise lr.ParameterError(
                "Since S.shape[-2] is {}, "
                "frame_length is expected to be {} or {}; "
                "found {}".format(
                    S.shape[-2], S.shape[-2] * 2 - 2, S.shape[-2] * 2 - 1, frame_length
                )
            )

        # power spectrogram
        x = np.abs(S) ** 2

        # Adjust the DC and sr/2 component
        x[..., 0, :] *= 0.5
        if frame_length % 2 == 0:
            x[..., -1, :] *= 0.5

        # Calculate power
        power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length ** 2
    else:
        raise lr.ParameterError("Either `y` or `S` must be input.")

    return np.sqrt(power)
def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, win_length=None, window="hann",
                      center=True, pad_mode="constant" ):
    S, n_fft = lr.core.spectrum._spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode)
    
    if not np.isrealobj(S):
        raise lr.ParameterError(
            "Spectral centroid is only defined " "with real-valued input"
        )
        
    elif np.any(S < 0):
        raise lr.ParameterError(
            "Spectral centroid is only defined " "with non-negative energies"
        )
        
    # Compute the center frequencies of each bin
    if freq is None:
        freq = lr.fft_frequencies(sr=sr, n_fft=n_fft)
        
    if freq.ndim == 1:
        # reshape for broadcasting
        freq = util.expand_to(freq, ndim=S.ndim, axes=-2)
        
        # Column-normalize S
        
    return np.sum(freq * util.normalize(S, norm=1, axis=-2), axis=-2, keepdims=True)

def spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window="hann", center=True,
    pad_mode="constant", freq=None, centroid=None, norm=True, p=2 ):
    S, n_fft = lr.core.spectrum._spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    
    if not np.isrealobj(S):
        raise lr.ParameterError(
            "Spectral bandwidth is only defined " "with real-valued input"
        )
    elif np.any(S < 0):
        raise lr.ParameterError(
            "Spectral bandwidth is only defined " "with non-negative energies"
        )

    # centroid or center?
    if centroid is None:
        centroid = spectral_centroid(
            y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq
        )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = lr.fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        deviation = np.abs(
            np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1)
        )
    else:
        deviation = np.abs(freq - centroid)

    # Column-normalize S
    if norm:
        S = util.normalize(S, norm=1, axis=-2)

    return np.sum(S * deviation ** p, axis=-2, keepdims=True) ** (1.0 / p)
def spectral_rolloff(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window="hann", center=True,
    pad_mode="constant", freq=None, roll_percent=0.85 ):
    if not 0.0 < roll_percent < 1.0:
        raise lr.ParameterError("roll_percent must lie in the range (0, 1)")

    S, n_fft = lr.core.spectrum._spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    if not np.isrealobj(S):
        raise lr.ParameterError(
            "Spectral rolloff is only defined " "with real-valued input"
        )
    elif np.any(S < 0):
        raise lr.ParameterError(
            "Spectral rolloff is only defined " "with non-negative energies"
        )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = lr.fft_frequencies(sr=sr, n_fft=n_fft)

    # Make sure that frequency can be broadcast
    if freq.ndim == 1:
        # reshape for broadcasting
        freq = util.expand_to(freq, ndim=S.ndim, axes=-2)

    total_energy = np.cumsum(S, axis=-2)
    # (channels,freq,frames)

    threshold = roll_percent * total_energy[..., -1, :]

    # reshape threshold for broadcasting
    threshold = np.expand_dims(threshold, axis=-2)

    ind = np.where(total_energy < threshold, np.nan, 1)

    return np.nanmin(ind * freq, axis=-2, keepdims=True)

def zero_crossing_rate(y, *, frame_length=2048, hop_length=512, center=True, **kwargs):
    # check if audio is valid
    util.valid_audio(y, mono=False)

    if center:
        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode="edge")

    y_framed = util.frame(y, frame_length=frame_length, hop_length=hop_length)

    kwargs["axis"] = -2
    kwargs.setdefault("pad", False)

    crossings = lr.zero_crossings(y_framed, **kwargs)

    return np.mean(crossings, axis=-2, keepdims=True)

def mfccc( y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm="ortho", lifter=0):
    if S is None:
        # multichannel behavior may be different due to relative noise floor differences between channels
        S = power_to_db(lr.feature.melspectrogram(y=y, sr=sr))

    M = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]

    if lifter > 0:
        # shape lifter for broadcasting
        LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)
        LI = util.expand_to(LI, ndim=S.ndim, axes=-2)

        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise lr.ParameterError(
            "MFCC lifter={} must be a non-negative number".format(lifter)
        )

def writeCsv(data):
    file = open("data.csv", 'a', newline='')
    writer = csv.writer(file)
    writer.writerow(data.split(","))
    file.close()

def feature_extraction(file_path):
    # iterate through all file
        to_append =[]
        header = 'filename rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 40):
            header += f' mfcc{i}'
        header = header.split()
        file = open("data.csv", 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        file.close()
        y, sr = lr.load(file_path, mono=True, duration=30)
            # remove leading and trailing silence
        y, index = lr.effects.trim(y)
        # chroma_stft = lr.feature.chroma_stft(y=y, sr=sr)
        rmse = rms(y=y)
        #rmse = lr.feature.rms(y=y)
        spec_cent = spectral_centroid(y=y, sr=sr)
        #spec_cent = lr.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = spectral_bandwidth(y=y, sr=sr)
        #spec_bw = lr.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = spectral_rolloff(y=y, sr=sr)
        #rolloff = lr.feature.spectral_rolloff(y=y, sr=sr)
        zcr = zero_crossing_rate(y)
        #zcr = lr.feature.zero_crossing_rate(y)
        mfcc = mfccc(y=y, sr=sr,n_mfcc=39)
        #mfcc = lr.feature.mfcc(y=y, sr=sr,n_mfcc=39)
        to_append = f'{file},{np.mean(rmse)},{np.mean(spec_cent)},{np.mean(spec_bw)},{np.mean(rolloff)},{np.mean(zcr)}'
        for e in mfcc:
            to_append += f',{np.mean(e)}'
        
        writeCsv(to_append)
        return to_append


def feature_extraction_array(file_path):
    to_append =[]
    # iterate through all file
    y, sr = lr.load(file_path, mono=True, duration=30)
        # remove leading and trailing silence
    y, index = lr.effects.trim(y)
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    #rmse = librosa.feature.rms(y=y)
    rmse = rms(y=y)
    #spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_cent = spectral_centroid(y=y, sr=sr)
    #spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw = spectral_bandwidth(y=y, sr=sr)
    #rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff = spectral_rolloff(y=y, sr=sr)
    #zcr = librosa.feature.zero_crossing_rate(y)
    zcr = zero_crossing_rate(y)
    #mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=39, n_fft=1024, hop_length=512)
    mfcc = mfccc(y=y, sr=sr,n_mfcc=39)
    to_append.append(np.mean(rmse))
    to_append.append(np.mean(spec_cent))
    to_append.append(np.mean(spec_bw))
    to_append.append(np.mean(rolloff))
    to_append.append(np.mean(zcr))
    for e in mfcc:
        to_append.append(np.mean(e))
    return to_append
    
    
def preProcessing(csvName):
    data = pd.read_csv(csvName)
    audioName = data['filename']
    speakerNumber = []
    for i in range(len(audioName)):
        speakerLetter = audioName[i][0]
        if speakerLetter == "A":
            speakerLetter = 0
        elif speakerLetter =="H":
            speakerLetter=1
        elif speakerLetter == "M":
            speakerLetter = 2
        elif speakerLetter =="Y":
            speakerLetter = 3
        else:
            speakerLetter = 4
        speakerNumber.append(speakerLetter)

    data = data.drop(['filename'],axis=1)

    print(data.tail())
    
    return data, speakerNumber


def plot_melspectrogram(file_name):
    audio,sfreq = lr.load(file_name)
    melspectrogram = lr.feature.melspectrogram(y=audio, sr=sfreq)
    fig=plt.figure(figsize=(25,10))
    img=librosa.display.specshow(melspectrogram,x_axis='time',y_axis='mel',sr=sfreq)
    return img,fig

def apply_model(features_list):
    loaded_model = pickle.load(open('./Model.sav', 'rb'))
    x_pre = np.array(features_list)
    x_pre = x_pre.reshape(1,-1)
    prediction=loaded_model.predict(x_pre)
    return prediction
