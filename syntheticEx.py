####################################################################################################
#                                          syntheticEx.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 10/03/21                                                                                #
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

from classicMUSIC import ULA_action_vector
from broadbandMUSIC import ULA_broad_action_vector


# set random seed
np.random.seed(42)


#********************#
#   initialization   #
#********************#
d = 5   # number of sources
m = 8   # number of array elements
snr = 5   # signal to noise ratio

mean_signal_power = 0
var_signal_power = 1

mean_noise = 0
var_noise = 1

doa = np.pi * (np.random.rand(d) - 1/2)   # random source directions in [-pi/2, pi/2]
p = np.sqrt(1) * (np.random.randn(d) + np.random.randn(d) * 1j)    # random source powers

array = np.linspace(0, m, m, endpoint=False)   # array element positions

# angles = np.array((np.linspace(- np.pi/2, np.pi/2, 360, endpoint=False),))   # angle continuum
# angles = np.array((np.linspace(0, 2 * np.pi, 360, endpoint=False),))   # angle continuum
angles = np.array([np.linspace(0, 2 * np.pi, 360, endpoint=False), np.linspace(-np.pi/2, 0, 360, endpoint=False)])


snapshots = 200


#***********************#
#   construct signals   #
#***********************#
def construct_signal(thetas):
    """
        Construct a signal with the given initializations.

        @param thetas -- The doa angles of the sources.

        @returns -- The measurement vector.
    """
    d = len(thetas)
    signal = np.sqrt(var_signal_power) * (10 ** (snr / 10)) * \
             (np.random.randn(d, snapshots) + 1j * np.random.randn(d, snapshots)) + mean_signal_power
    A = np.array([ULA_action_vector(array, thetas[j]) for j in range(d)])
    noise = np.sqrt(var_noise) * (np.random.randn(m, snapshots) + 1j *
                                  np.random.randn(m, snapshots)) + mean_noise

    return np.dot(A.T, signal) + noise, signal


#********************************#
#   construct coherent signals   #
#********************************#
def construct_coherent_signal(thetas):
    """
        Construct a coherent signal with the given initializations.

        @param thetas -- The doa angles of the sources.

        @returns -- The measurement vector.
    """
    d = len(thetas)
    signal = np.sqrt(var_signal_power) * (10 ** (snr / 10)) * \
             (np.random.randn(1, snapshots) + 1j * np.random.randn(1, snapshots)) + mean_signal_power

    # all signals receive same amplitude and phase...
    signal = np.repeat(signal, d, axis=0)

    A = np.array([ULA_action_vector(array, thetas[j]) for j in range(d)])
    noise = np.sqrt(var_noise) * (np.random.randn(m, snapshots) + 1j *
                                  np.random.randn(m, snapshots)) + mean_noise

    return np.dot(A.T, signal) + noise, signal


#********************#
#   create dataset   #
#********************#
def create_dataset(name, size, coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, d))
    for i in tqdm(range(size)):
        thetas = np.pi * (np.random.rand(d) - 1/2)  # random source directions
        if coherent: X[i] = construct_coherent_signal(thetas)[0]
        else: X[i] = construct_signal(thetas)[0]
        Thetas[i] = thetas

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas


#**************************#
#   create mixed dataset   #
#**************************#
def create_mixed_dataset(name, first, second, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param first -- The path/name of the first dataset to be mixed with...
        @param second -- The path/name of the second dataset.
        @param save -- If true the dataset is saved to filename.
    """
    hf1 = h5py.File(first + '.h5', 'r')
    hf2 = h5py.File(second + '.h5', 'r')

    dataX1 = np.array(hf1.get('X'))
    dataY1 = np.array(hf1.get('Y'))

    dataX2 = np.array(hf2.get('X'))
    dataY2 = np.array(hf2.get('Y'))

    dataX = np.concatenate((dataX1, dataX2), axis=1)
    dataY = np.concatenate((dataY1, dataY2), axis=1)

    dataX, dataY = utils.shuffle(dataX, dataY)

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=dataX)
        hf.create_dataset('Y', data=dataY)
        hf.close()

    return dataX, dataY


#*******************************#
#   create resolution dataset   #
#*******************************#
def create_res_cap_dataset(name, size, space, coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves
        used for testing the resolution capabilities of DoA estimation algorithms
        by creating two closely spaced signals.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param space -- The distance of the closely spaced signals.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, 2))
    for i in tqdm(range(size)):
        theta = np.pi * (np.random.rand(1) - 1/2)   # random source direction
        thetas = [theta, (((theta + space) + np.pi/2) % np.pi) - np.pi/2]
        if coherent: X[i] = construct_coherent_signal(thetas)[0]
        else: X[i] = construct_signal(thetas)[0]
        Thetas[i] = thetas

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas


#********************************#
#   create angle-shift dataset   #
#********************************#
def create_ang_shift_dataset(name, size, angle, coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves
        used for testing the performance of DoA estimation algorithms when the
        data is shifted by a certain angle.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param angle -- The angle-shift in degree.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, d))
    for i in tqdm(range(size)):
        thetas = np.pi * (np.random.rand(d) - 1 / 2) + angle * np.pi / 180
        if coherent: X[i] = construct_coherent_signal(thetas)[0]
        else: X[i] = construct_signal(thetas)[0]
        Thetas[i] = (thetas + np.pi / 2) % np.pi - np.pi / 2

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas


#***************************************#
#   create dataset with large variety   #
#***************************************#
def create_complete_dataset(name, size, num_sources=[d], coherent=False, save=True):
    """
        Creates dataset of given size with the above initializations and saves.

        @param name -- The name (an path) of the file of the dataset.
        @param size -- The size of the dataset.
        @param num_sources -- The number of sources as a list.
        @param coherent -- If true, the signals are coherent.
        @param save -- If true, the dataset is saved to filename.
    """
    X = np.zeros((size, m, snapshots)) + 1j * np.zeros((size, m, snapshots))
    Thetas = np.zeros((size, m))
    for i in tqdm(range(size)):
        num = num_sources[i % len(num_sources)]   # create equal sized sets for each num. of sources
        thetas = np.pi * (np.random.rand(num) - 1/2)   # random source direction
        if coherent: X[i] = construct_coherent_signal(thetas)[0]
        else: X[i] = construct_signal(thetas)[0]
        Thetas[i] = np.pad(thetas, (0, m - num), 'constant', constant_values=np.pi)

    if save:
        hf = h5py.File('data/' + name + '.h5', 'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('Y', data=Thetas)
        hf.close()

    return X, Thetas


if __name__ == "__main__":
    # create_mixed_dataset('m_d2_l200_snr10_2k', first='data/d2_l200_snr10_1k',
    #                                            second='data/c_d2_l200_snr10_1k')

    # create_res_cap_dataset('m8/res0.20_l200_snr10_10k', 10000, 0.20)

    # create_ang_shift_dataset('m8/phase-25_d5_l200_snr10_1k', 1000, -25, coherent=False)

    create_dataset('m8/d5_l200_snr10_1k', 1000)

    # create_complete_dataset('m8/d2-5_l200_snr10_100k_c', 100000, num_sources=[2, 3, 4, 5], coherent=True)
