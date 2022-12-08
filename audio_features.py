import os
import pandas as pd
import librosa as lr
import librosa.display 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from librosa import power_to_db , util
import scipy

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
    
    