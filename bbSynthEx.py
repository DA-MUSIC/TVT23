####################################################################################################
#                                          syntheticEx.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/08/21                                                                                #
#                                                                                                  #
# Purpose: Synthetic examples to test correctness and performance of algorithms estimating         #
#          directions of arrival (DoA) of multiple signals.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy import signal
from sklearn import utils
from tqdm import tqdm

from broadbandMUSIC import ULA_broad_action_vector


# set random seed
np.random.seed(42)


#********************#
#   initialization   #
#********************#

d = 5   # number of sources
m = 8   # number of array elements
snr = 10   # signal to noise ratio

# frequency range of interest
fs = np.linspace(0, 1000, 1000, endpoint=False)

# sampling frequency
fSamp = 2000

# distance between array elements = min wavelength / 2
dist = (3e8 / np.max(fs) + 1) / 2
# dist = 0.03

doa = np.pi * (np.random.rand(d) - 1/2)   # random source directions in [-pi/2, pi/2]
p = np.sqrt(1) * (np.random.randn(d) + np.random.randn(d) * 1j)    # random source powers

array = np.linspace(0, m, m, endpoint=False)   # array element positions
angles = np.array((np.linspace(- np.pi/2, np.pi/2, 360, endpoint=False),))   # angle continuum
# angles = np.array((np.linspace(0, 2 * np.pi, 720, endpoint=False),))   # angle continuum

snapshots = 200  # number of time samples
t = np.arange(0, 1, 1/fSamp)   # discrete time events


#***********************#
#   construct signals   #
#***********************#
def construct_broad_signal(thetas, fcs):
    """
        Construct a signal with the given initializations.

        @param thetas -- The doa angles of the sources.
        @param fcs -- The carrier frequencies of the sources.

        @returns -- The measurement vector.
    """
    d = len(thetas)

    signal = np.zeros((d, len(t))) + 1j * np.zeros((d, len(t)))

    for i, fc in enumerate(fcs):
        # random amplitude and phase
        amp = np.sqrt(2)/2 * (np.random.randn() + 1j * np.random.randn())

        # construct signal with given carrier frequency
        signal[i] =  (10 ** (snr / 10)) * amp * np.exp(1j * 2 * np.pi * fc * t)

    # noise with random amplitude and phase
    noise = np.sqrt(2)/2 * (np.random.randn(m, len(t)) + 1j * np.random.randn(m, len(t)))

    # transform to frequency domain
    signal = np.fft.fft(signal)
    noise = np.fft.fft(noise)

    X = []
    for i in range(int(fSamp)):

        # mapping from index i to frequency f
        if i > int(fSamp) // 2: f = - int(fSamp) + i
        else: f = i

        # construct signal
        A = np.array([ULA_broad_action_vector(array, thetas[j], f, dist) for j in range(d)])
        X.append(np.dot(A.T, signal[:, i]) + noise[:, i])

    # plt.plot(range(int(fSamp)), np.abs(np.array(X)[:, 0]))

    return np.fft.ifft(np.array(X).T, axis=1, n=snapshots)[:, :snapshots], signal


#***********************#
#   construct signals   #
#***********************#
def construct_ofdm_signal(thetas, fcs):
    """
        Construct an OFDM signal with the given initializations.

        @param thetas -- The doa angles of the sources.
        @param fcs -- The carrier frequencies of the sources.

        @returns -- The measurement vector.
    """
    d = len(thetas)
    numSub = 1000   # number of subcarriers per signal

    signal = np.zeros((d, len(t))) + 1j * np.zeros((d, len(t)))

    for i in range(d):
        for j in range(numSub):
            # random amplitude and phase
            amp = np.sqrt(2)/2 * (np.random.randn() + 1j * np.random.randn())

            # construct signal by summing subcarriers
            signal[i] += amp * np.exp(1j * 2 * np.pi * j * len(fs) * t / numSub)

        signal[i] *= (10 ** (snr / 10)) * (1/numSub)

    # noise with random amplitude and phase
    noise = np.sqrt(2)/2 * (np.random.randn(m, len(t)) + 1j * np.random.randn(m, len(t)))

    # transform to frequency domain
    signal = np.fft.fft(signal)
    noise = np.fft.fft(noise)

    X = []
    for i in range(int(fSamp)):

        # mapping from index i to frequency f
        if i > int(fSamp) // 2: f = - int(fSamp) + i
        else: f = i

        # construct signal
        A = np.array([ULA_broad_action_vector(array, thetas[j], f, dist) for j in range(d)])
        X.append(np.dot(A.T, signal[:, i]) + noise[:, i])

    return np.fft.ifft(np.array(X).T, axis=1)[:, :snapshots]


#***********************#
#   construct signals   #
#***********************#
def construct_mod_ofdm_signal(thetas, fcs):
    """
        Construct an OFDM signal with the given initializations.

        @param thetas -- The doa angles of the sources.
        @param fcs -- The carrier frequencies of the sources.

        @returns -- The measurement vector.
    """
    d = len(thetas)
    numSub = 10   # number of subcarriers per signal
    bw = 100   # bandwidth of subcarriers

    signal = np.zeros((d, len(t))) + 1j * np.zeros((d, len(t)))

    for i, fc in enumerate(fcs):
        for j in range(numSub):
            # random amplitude and phase
            amp = np.sqrt(2) / 2 * (np.random.randn() + 1j * np.random.randn())

            # construct signal by summing subcarriers
            signal[i] += amp * np.exp(1j * 2 * np.pi * j * bw * t / numSub)

        # modulate signal with given carrier frequency
        signal[i] *= (10 ** (snr / 10)) * (1/numSub) * np.exp(1j * 2 * np.pi * fc * t)

    # noise with random amplitude and phase
    noise = np.sqrt(2) / 2 * (np.random.randn(m, len(t)) + 1j * np.random.randn(m, len(t)))

    # transform to frequency domain
    signal = np.fft.fft(signal)
    noise = np.fft.fft(noise)

    X = []
    for i in range(int(fSamp)):

        # mapping from index i to frequency f
        if i > int(fSamp) // 2: f = - int(fSamp) + i
        else: f = i

        # construct signal
        A = np.array([ULA_broad_action_vector(array, thetas[j], f, dist) for j in range(d)])
        X.append(np.dot(A.T, signal[:, i]) + noise[:, i])

    return np.fft.ifft(np.array(X).T, axis=1)[:, :snapshots]



#***************************************#
#   create dataset with large variety   #
#***************************************#
def create_broadband_dataset(name, size, num_sources=[d], coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param num_sources -- The number of sources as a list.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    # Thetas = np.zeros((size, m))
    Thetas = np.zeros((size, d))
    for i in tqdm(range(size)):
        num = num_sources[i % len(num_sources)]   # create equal sized sets for each num. of sources

        thetas = np.pi * (np.random.rand(num) - 1/2)   # random source direction
        fcs = np.random.choice(fs, d)  # random carrier freq in range fs

        if coherent: print("ERROR - not implemented yet!")
        # else: X[i] = construct_broad_signal(thetas, fcs)
        else: X[i] = construct_ofdm_signal(thetas, fcs)
        # else: X[i] = construct_mod_ofdm_signal(thetas, fcs)
        # Thetas[i] = np.pad(thetas, (0, m - num), 'constant', constant_values=np.pi)
        Thetas[i] = thetas

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas


if __name__ == "__main__":

    create_broadband_dataset('m8/bb0-100k(100k)_fs200k_ofdm1k_d2_l200k_snr10_10k', 10000, num_sources=[5], coherent=False)
