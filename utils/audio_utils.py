import librosa
import numpy as np

MAX_LEN = 100
N_MFCC = 40

def preprocess_audio(uploaded_file):
    y, sr = librosa.load(uploaded_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    mfcc = mfcc.T
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc
