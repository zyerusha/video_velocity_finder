from scipy.signal import butter, lfilter, filtfilt
import numpy as np


class Filters:
    def __init__(self):
        return None

    [staticmethod]

    def ButterLowpass(data, cutoff, fs, order):
        nyq = 0.5 * fs  # Nyquist Freq
        w = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, w, btype='low', analog=False)
        # print(f"a: {a}, b: {b} data_len: {len(data)}")
        y = filtfilt(b, a, data)

        return y

    [staticmethod]

    def ButterHighpass(data, cutoff, fs, order):
        nyq = 0.5 * fs  # Nyquist Freq
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y

    [staticmethod]

    def ButterBandpass(data, cutoff_low, cutoff_high, fs, order):
        nyq = 0.5 * fs  # Nyquist Freq
        normal_cutoff_low = cutoff_low / nyq
        normal_cutoff_high = cutoff_high / nyq
        # Get the filter coefficients
        b, a = butter(order, [normal_cutoff_low, normal_cutoff_high], btype='band', analog=False)
        y = filtfilt(b, a, data)
        return y

    [staticmethod]

    def SignalToNoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)
