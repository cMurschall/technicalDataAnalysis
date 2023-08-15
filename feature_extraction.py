import scipy
import numpy as np
import tsfel


def calc_statistical_features(signal):
    """This function computes the statistical features of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which statistical features are computed

    Returns
    -------
    features: nd-array
        Statistical features

    """

    features = np.array([])

    # Mean
    features = np.append(features, np.mean(signal))

    # Median
    features = np.append(features, np.median(signal))

    # Standard deviation
    features = np.append(features, np.std(signal))

    # Root mean square
    features = np.append(features, np.sqrt(np.mean(np.square(signal))))

    # Interquartile range
    features = np.append(features, scipy.stats.iqr(signal))

    # Skewness
    features = np.append(features, scipy.stats.skew(signal))

    # Kurtosis
    features = np.append(features, scipy.stats.kurtosis(signal))

    # Mean absolute deviation
    features = np.append(features, np.mean(np.abs(signal - np.mean(signal))))

    # Signal magnitude area
    features = np.append(features, np.sum(np.abs(signal)))

    # Energy
    features = np.append(features, np.sum(np.square(signal)))

    # Entropy
    # features = np.append(features, scipy.stats.entropy(signal))

    return features


def calc_spectral_features(signal):
    """This function computes the spectral features of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which spectral features are computed

    Returns
    -------
    features: nd-array
        Spectral features

    """

    features = np.array([])

    # Spectral centroid
    features = np.append(features, scipy.stats.mstats.moment(signal, moment=1))

    # Spectral spread
    features = np.append(features, scipy.stats.mstats.moment(signal, moment=2))

    # Spectral entropy
    # features = np.append(features, scipy.stats.entropy(signal))

    # Spectral energy
    features = np.append(features, np.sum(np.square(signal)))

    # Spectral flux
    features = np.append(features, np.sum(np.square(np.diff(signal))))

    # Spectral roll-off
    features = np.append(features, scipy.stats.mstats.moment(signal, moment=3))

    # Spectral flatness
    features = np.append(features, scipy.stats.mstats.moment(signal, moment=4))

    # Spectral crest factor
    features = np.append(features, np.max(signal) / np.sum(signal))

    # Spectral slope
    features = np.append(features, scipy.stats.linregress(np.arange(len(signal)), signal)[0])

    # Spectral decrease
    features = np.append(features, np.sum(np.diff(signal) < 0))

    # Spectral increase
    features = np.append(features, np.sum(np.diff(signal) > 0))

    # Spectral variation
    features = np.append(features, np.sum(np.diff(signal) != 0))

    return features


def calc_temporal_features(signal):
    """This function computes the temporal features of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which temporal features are computed

    Returns
    -------
    features: nd-array
        Temporal features

    """

    features = np.array([])

    # Zero crossing rate
    features = np.append(features, np.sum(np.diff(signal > 0)))

    # Slope sign change
    features = np.append(features, np.sum(np.diff(np.sign(np.diff(signal)))) / len(signal))

    # Mean absolute diff
    features = np.append(features, np.mean(np.abs(np.diff(signal))))

    # Peak to peak distance
    features = np.append(features, np.max(signal) - np.min(signal))

    # Waveform length
    features = np.append(features, np.sum(np.abs(np.diff(signal))))

    # Willison amplitude
    features = np.append(features, np.sum(np.abs(np.diff(signal) > 0.0001)))

    return features


def extract(signal):
    """This function extracts the features from the signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which features are extracted

    Returns
    -------
    features: nd-array
        Extracted features

    """

    features = np.array([])

    # Statistical
    features = np.append(features, calc_statistical_features(signal))

    # Spectral
    features = np.append(features, calc_spectral_features(signal))

    # Temporal
    # features = np.append(features, calc_temporal_features(signal))

    return features


def get_feature_size():
    """This function returns the size of the feature vector.

    Returns
    -------
    size: int
        Size of the feature vector

    """

    return extract(np.random.rand(100)).shape


def calc_fft(signal, sampling_frequency):
    """This functions computes the fft of a signal.

    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    sf : int
        Sampling frequency

    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)

    """

    fmag = np.abs(np.fft.rfft(signal))
    len_signal = len(signal[-1, ...]) if len(np.shape(signal)) > 1 else len(signal)
    f = np.fft.rfftfreq(len_signal, d=1 / sampling_frequency)

    return f.copy(), fmag.copy()
