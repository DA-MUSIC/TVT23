####################################################################################################
#                                           beamformer.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 22/04/21                                                                                #
#                                                                                                  #
# Purpose: Implementation of the purely model-based classical Beamformer algorithm.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import warnings

from scipy import linalg

from broadbandMUSIC import UCA_broad_action_vector
from classicMUSIC import UCA_action_vector
from bbSynthEx import *


# shut up casting warnings
warnings.simplefilter("ignore")


#******************************#
#   the Beamformer algorithm   #
#******************************#
def beamformer(incident, array, continuum, slow, sources=None):
    """
        The classical Beamformer algorithm calculates a spatial spectrum, which is used to
        estimate the directions of arrival of the incident signals by finding its d peaks.

        @param incident -- The measured waveforms (= incident signals and noise).
        @param array -- Holds the positions of the array elements.
        @param continuum -- The continuum of all possible mode vectors
        @param sources -- The number of signal sources.

        @returns -- The d locations of the spatial spectrum peaks.
    """
    # calculate EVD of covariance matrix
    covariance = np.cov(incident)

    # calculate spatial spectrum
    numSamples = continuum.shape[1]
    spectrum = np.zeros(numSamples)
    spectrum2D = np.zeros((numSamples, numSamples))
    for i in range(numSamples):
        for j in range(numSamples):
            # establish array steering vector
            # a = ULA_action_vector(array, continuum[0, i])

            a = build_steering_vect(array, continuum[0, i], continuum[1, j], slow)
            spectrum2D[j, i] = (a.conj().transpose() @ covariance @ a) / linalg.norm(a)**2

        a = build_steering_vect(array, continuum[0, i], slow=slow)

        spectrum[i] = (a.conj().transpose() @ covariance @ a) / linalg.norm(a) ** 2

    DoAsMUSIC, _ = signal.find_peaks(spectrum)

    # only keep d largest peaks...
    if sources: DoAsMUSIC = DoAsMUSIC[np.argsort(spectrum[DoAsMUSIC])[-sources:]]

    # or give all peak locations in descending order
    else: DoAsMUSIC = DoAsMUSIC[np.argsort(- spectrum[DoAsMUSIC])]

    return DoAsMUSIC, spectrum, spectrum2D


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


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect(r, azimuth, elev=None):
    wavelength = 250
    if elev: theta = elev
    else: theta = -0.1

    k = np.array([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])

    return np.exp(- 1j * 2 * np.pi / wavelength * np.dot(r, k))


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect(r, azimuth, elev=None, slow=None):
    f = 1
    if elev: theta = elev
    else: theta = - np.pi/4

    if slow: v = 1/slow
    else: v = 1

    k = np.array([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])

    return np.exp(- 1j * 2 * np.pi * f / v * np.dot(r, k))


# #************************************************#
# #   build steering vectors from r and azimuth    #
# #************************************************#
# def build_steering_vect(r, azimuth, elev=None):
#     wavelength = 23.157793209876544
#     if elev: theta = elev
#     else: theta = -0.5526148744127046
#
#     k = np.array([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])
#
#     return np.exp(- 1j * 2 * np.pi / wavelength * np.dot(r, k))
#
#
# #************************************************#
# #   build steering vectors from r and azimuth    #
# #************************************************#
# def build_steering_vect(r, azimuth, elev=None):
#     wavelength = azimuth
#     if elev: theta = elev
#     else: theta = - np.pi/4
#
#     azimuth = 5.81153364
#
#     k = np.array([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])
#
#     return np.exp(- 1j * 2 * np.pi / wavelength * np.dot(r, k))