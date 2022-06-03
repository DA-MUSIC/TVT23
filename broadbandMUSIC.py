####################################################################################################
#                                         broadbandMUSIC.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 03/08/21                                                                                #
#                                                                                                  #
# Purpose: Implementation of the purely model-based broadband MUSIC algorithm.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics
import tensorflow as tf
import warnings

from scipy import linalg
from scipy import signal


# shut up casting warnings
warnings.simplefilter("ignore")

# set random seed
np.random.seed(42)


# frequency range of interest
fs = np.linspace(0, 2000 -1, 2000 -1, endpoint=False)

# sampling frequency
fSamp = 40
# fSamp = 160

# distance between array elements = min wavelength / 2
dist = (3e8 / np.max(fs) + 1) / 2
# dist = 0.03

c = 10000


#***********************************#
#   the broadband MUSIC algorithm   #
#***********************************#
def broadMUSIC(incident, array, continuum, slow, sources=None, fs=fs, NUMS=10):
    """
        The classic MUSIC algorithm calculates the spatial spectrum, which is used to estimate
        the directions of arrival of the incident signals by finding its d peaks.

        @param incident -- The measured waveforms (= incident signals and noise).
        @param array -- Holds the positions of the array elements.
        @param continuum -- The continuum of all possible mode vectors
        @param sources -- The number of signal sources (optional).

        @returns -- The d locations of the spatial spectrum peaks.
    """
    # incident = np.fft.fft(incident, axis=1, n=fSamp)
    incident = np.fft.fft(incident, axis=1)

    # get frequencies from measurements
    freqs = np.fft.fftfreq(incident.shape[1], d=1.0/fSamp)

    # calculate spatial spectrum
    numSamples = continuum.shape[1]
    spectra = np.zeros((NUMS, numSamples))
    spectra2D = np.zeros((NUMS, numSamples, numSamples))
    for i in range(NUMS):
        ind = int(np.min(fs)) + i * len(fs) // NUMS

        # calculate EVD of covariance matrix
        covariance = np.cov(incident[:, ind:ind + len(fs) // NUMS])
        eigenvalues, eigenvectors = linalg.eig(covariance)

        if sources:  # number of sources known
            d = sources
        else:
            n = cluster(eigenvalues).shape[0]  # estimate multiplicity of smallest eigenvalue...
            d = array.shape[0] - n  # and get number of signal sources

        # the noise matrix
        En = eigenvectors[:, d:]

        spectrum = np.zeros(numSamples)
        spectrum2D = np.zeros((numSamples, numSamples))
        for j in range(numSamples):
            for k in range(numSamples):
                # a = build_steering_vect(array, continuum[0][j], freqs[(ind + len(fs) // NUMS)], continuum[1][k], slow=slow)
                a = build_steering_vect(array, continuum[0][j], 1, continuum[1][k], slow=slow)
                spectrum2D[k, j] = 1. / (a.conj().transpose() @ En @ En.conj().transpose() @ a)


            # establish array steering vector
            # a = ULA_broad_action_vector(array, continuum[0][j], ind + len(fs)//NUMS - 1, dist)

            # a = build_steering_vect(array, continuum[0][j], freqs[(ind + len(fs) // NUMS)], slow=slow)
            a = build_steering_vect(array, continuum[0][j], 1, slow=slow)
            spectrum[j] = 1./(a.conj().transpose() @ En @ En.conj().transpose() @ a)

        spectra[i] = spectrum
        spectra2D[i] = spectrum2D

    # average spectra to one spectrum
    spectrum = np.mean(spectra, axis=0)
    spectrum2D = np.mean(spectra2D, axis=0)

    DoA, _ = signal.find_peaks(spectrum)

    # only keep d largest peaks
    DoA = DoA[np.argsort(spectrum[DoA])[-d:]]

    return DoA, spectrum, spectra, spectrum2D


#*******************************#
#   cluster small eigenvalues   #
#*******************************#
def cluster(evs):
    """
        Estimates multiplicity of smallest eigenvalue.

        @param evs -- The eigenvalues in descending order.

        @returns -- The eigenvalues similar or equal to the smallest eigenvalue.
    """
    # simplest clustering method: with threshold
    threshold = 0.4
    return evs[np.where(abs(evs) < abs(evs[-1]) + threshold)]


#*******************************************#
#   uniform linear array steering vector    #
#*******************************************#
def ULA_broad_action_vector(array, theta, f, spacing):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.
        @param f -- Holds the frequency of the signals.
        @param spacing -- The distance between the array elements.

        @returns -- The action vector.
    """
    return np.exp(- 1j * 2 * np.pi * f * spacing * array * np.sin(theta) / 3e8)


#*********************************************#
#   uniform circular array steering vector    #
#*********************************************#
def UCA_broad_action_vector(array, theta, f, radius):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.
        @param f -- Holds the frequency of the signals.
        @param radius -- The radius of the given UCA.

        @returns -- The action vector.
    """
    return np.exp(- 1j * 2 * np.pi * radius * f *
                  np.sin(theta - 2 * np.pi * array / len(array)) / 343)


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect(r, azimuth, f, elev=None):
    if elev: theta = elev
    else: theta = - np.pi/4

    k = 2 * np.pi * f / c * np.array([np.sin(theta) * np.cos(azimuth),
                                      np.sin(theta) * np.sin(azimuth), np.cos(theta)])

    return np.exp(- 1j * np.dot(r, k))


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect(r, azimuth, f, elev=None, slow=None):
    if elev: theta = elev
    else: theta = - np.pi/4

    if slow: v = 1/slow
    else: v = 1

    k = 2 * np.pi * f / v * np.array([np.sin(theta) * np.cos(azimuth),
                                      np.sin(theta) * np.sin(azimuth), np.cos(theta)])

    return np.exp(- 1j * np.dot(r, k))


# #************************************************#
# #   build steering vectors from r and azimuth    #
# #************************************************#
# def build_steering_vect(r, azimuth, f, elev=None):
#     wavelength = 44.21026234567901
#     if elev: theta = elev
#     else: theta = -1.1443784535750041
#
#     k = np.array([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])
#
#     return np.exp(- 1j * 2 * np.pi / wavelength * np.dot(r, k))
#
#
# #************************************************#
# #   build steering vectors from r and azimuth    #
# #************************************************#
# def build_steering_vect(r, azimuth, f, elev=None):
#     wavelength = azimuth
#     if elev: theta = elev
#     else: theta = - np.pi/4
#
#     azimuth = 5.81153364
#
#     k = np.array([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])
#
#     return np.exp(- 1j * 2 * np.pi / wavelength * np.dot(r, k))