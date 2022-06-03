####################################################################################################
#                                            augMUSIC.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 23/03/21                                                                                #
#                                                                                                  #
# Purpose: Implementation of the augmented MUSIC algorithm. The augmentation obtains the           #
#          subspaces directly through a neural network from the measurments.                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import matplotlib.pyplot as plt
import numpy as np

from scipy import linalg
from scipy import signal

from tensorflow.keras.models import load_model


#***********************************#
#   the augmented MUSIC algorithm   #
#***********************************#
def augMUSIC(En_stack, array, continuum, sources):
    """
        The MUSIC algorithm calculates the spatial spectrum, which is used to estimate
        the directions of arrival of the incident signals by finding its d peaks.

        @param En_stack -- The estimated noise subspace (real and imag stacked).
        @param array -- Holds the positions of the array elements.
        @param continuum -- The continuum of all possible mode vectors
        @param sources -- The number of signal sources.

        @returns -- The d locations of the spatial spectrum peaks.
    """
    # the noise matrix
    En = En_stack[0, :array.shape[0]] + 1j * En_stack[0, array.shape[0]:]

    # calculate spatial spectrum
    numSamples = continuum.shape[1]
    spectrum = np.zeros(numSamples)
    for axis in continuum:
        for i in range(numSamples):
            # establish array steering vector
            a = ULA_action_vector(array, axis[i])
            spectrum[i] = 1./(a.conj().transpose() @ En @ En.conj().transpose() @ a)

    DoAsMUSIC, _ = signal.find_peaks(spectrum)

    # only keep d largest peaks
    DoAsMUSIC = DoAsMUSIC[np.argsort(spectrum[DoAsMUSIC])[-sources:]]

    return DoAsMUSIC, spectrum


#*******************************************#
#   uniform linear array steering vector    #
#*******************************************#
def ULA_action_vector(array, theta):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.

        @returns -- The action vector.
    """
    return np.exp(- 1j * np.pi * array * np.sin(theta))