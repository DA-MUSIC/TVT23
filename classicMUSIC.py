####################################################################################################
#                                          classicMUSIC.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 06/03/21                                                                                #
#                                                                                                  #
# Purpose: Implementation of the purely model-based MUSIC algorithm.                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import matplotlib.pyplot as plt
import numpy as np
import warnings

from scipy import linalg
from scipy import signal

# shut up casting warnings
warnings.simplefilter("ignore")


#*********************************#
#   the classic MUSIC algorithm   #
#*********************************#
def classicMUSIC(incident, array, continuum, slow, sources=None):
    """
        The classic MUSIC algorithm calculates the spatial spectrum, which is used to estimate
        the directions of arrival of the incident signals by finding its d peaks.

        @param incident -- The measured waveforms (= incident signals and noise).
        @param array -- Holds the positions of the array elements.
        @param continuum -- The continuum of all possible mode vectors
        @param sources -- The number of signal sources (optional).

        @returns -- The d locations of the spatial spectrum peaks.
    """
    # calculate EVD of covariance matrix
    covariance = np.cov(incident)
    eigenvalues, eigenvectors = linalg.eig(covariance)

    if sources:   # number of sources known
        d = sources
    else:
        n = cluster(eigenvalues).shape[0]   # estimate multiplicity of smallest eigenvalue...
        d = array.shape[0] - n   # and get number of signal sources

    # the noise matrix
    En = eigenvectors[:, d:]

    # calculate spatial spectrum
    numSamples = continuum.shape[1]
    spectrum = np.zeros(numSamples)
    spectrum2D = np.zeros((numSamples, numSamples))
    for i in range(numSamples):
        for j in range(numSamples):
            # establish array steering vector
            # a = ULA_action_vector(array, continuum[0, i])

            a = build_steering_vect(array, continuum[0, i], continuum[1, j], slow)
            spectrum2D[j, i] = 1./(a.conj().transpose() @ En @ En.conj().transpose() @ a)

        a = build_steering_vect(array, continuum[0, i], slow=slow)
        spectrum[i] = 1. / (a.conj().transpose() @ En @ En.conj().transpose() @ a)

    DoA, _ = signal.find_peaks(spectrum)

    # only keep d largest peaks
    DoA = DoA[np.argsort(spectrum[DoA])[-d:]]

    return DoA, spectrum, d, spectrum2D


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
    threshold = 1.25   # non-coherent
    # threshold = 0.1   # coherent
    return evs[np.where(abs(evs) < abs(evs[-1]) + threshold)]


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
    wavelength = 200
    azimuth = theta
    theta = np.pi / 2

    k = np.array([np.sin(theta) * np.cos(azimuth), np.sin(theta) * np.sin(azimuth), np.cos(theta)])
    r = np.array([[0, i * wavelength / 2, 0] for i in array])

 #    r = np.array([[   0.,            0. ,           0.        ],
 # [ 656.22986676, 1416.93947225,  -34.        ],
 # [  55.60325011, 1996.93023541,   68.        ],
 # [ 456.05800056,  220.24965948,  -21.        ],
 # [ 400.45471554,  367.08276575,   -3.        ],
 # [ 656.22986676,  212.90800416,  -54.        ],
 # [ 255.77499064,   80.75820848,  -10.        ],
 # [ 689.8142638 ,  631.38235687,   -9.        ],
 # [ 756.31582342,   22.02496595,  -88.        ],
 # [ 478.52172908,  212.90800416,  -53.        ],
 # [ 211.2923762,   513.91587196,   29.        ],
 # [ 945.58953105,  807.58208409,  -22.        ],
 # [1078.81512393,  528.59918258,  -69.        ],
 # [ 278.2387121,   836.94870528,  -97.        ],
 # [ 344.51762961,  234.9329701 ,   -3.        ],
 # [ 655.89557245,  741.5071864 ,   64.        ],
 # [ 300.36881472, 1145.29822734,   20.        ],
 # [2124.04759817, 1167.32319317,  -56.        ],
 # [2279.73722913,  183.5413829 ,    7.        ],
 # [1534.76268908, 1255.42305645,   60.        ],
 # [ 900.66077009, 1336.18126439,  -79.        ],
 # [1478.82226103,  168.85807227,  -78.        ],
 # [1312.12408919, 1248.08140118,   46.        ],
 # [1189.91061038, 1828.07216561,   19.        ]])

    return np.exp(- 1j * 2 * np.pi / wavelength * np.dot(r, k))

    # return np.exp(- 1j * np.pi * array * np.sin(theta))


#*********************************************#
#   uniform circular array steering vector    #
#*********************************************#
def UCA_action_vector(array, theta):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.

        @returns -- The action vector.
    """
    return np.exp(- 1j * np.pi * np.sin(theta - 2 * np.pi * array / len(array)))


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect(r, azimuth, elev=None):
    wavelength = 250
    if elev: theta = elev
    else: theta = - np.pi/4

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
#     wavelength = 18.42098765432099
#     if elev: theta = elev
#     else: theta = -1.3706409985488246
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