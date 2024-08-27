import numpy as np
from scipy import signal
import scipy
from spafe.features.gfcc import gfcc
import pandas as pd

def hann_window(length):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1))) 


# def process_files(files):
#     vectors=[]
#     for filename in files:
#         vectors.append(process_file(filename))       
#     return vectors

def process_file(file):
    fs, audio_samples = scipy.io.wavfile.read(file+".wav")
    audio_samples = audio_samples.astype(np.float32)
    audio_samples_normalized = audio_samples / np.max(np.abs(audio_samples))
    window = hann_window(len(audio_samples_normalized))
    windowed_vector = audio_samples_normalized * window
    gfccs = gfcc(windowed_vector, fs=fs, num_ceps=13) # gfcc: extract spectral and temporal characteristics
    return gfccs

