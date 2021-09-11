import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

def butter_lowpass(cutoff, fs, order):
    nyq = 5.1 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=5, fs=64, order=3, start_from=1000):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return pd.Series(y[start_from:])


def filter_signal(signal, original_sampling):
    result = scipy.stats.mstats.winsorize(signal, limits=[0.03, 0.03])

    result = butter_lowpass_filter(result, fs=original_sampling, start_from=0)

    result = np.array(result).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(result)
    result = scaler.transform(result)

    return result.reshape(1, -1)[0]
