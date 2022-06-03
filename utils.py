####################################################################################################
#                                             utils.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/04/21                                                                                #
#                                                                                                  #
# Purpose: Definitions of helpful functions.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import Regularizer

from scipy import signal
from scipy.stats import laplace
from syntheticEx import *
from bbSynthEx import fs, dist


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)
tf.random.set_seed(42)


#**********************************#
#   calculate the MUSIC spectrum   #
#**********************************#
def calculate_spectrum_ang(y_pred, ang='az', sv=None, calculateA=True):
    """
        Calculates the MUSIC spectrum according to P = 1 / (a^H En En^H a).

        @param y_pred -- The estimated noise space vectors.

        @returns -- The estimated spectrum.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]

    EnReal = Lambda(lambda y: y[:, :m])(y_pred)
    EnImag = Lambda(lambda y: y[:, m:])(y_pred)

    En = tf.cast(tf.dtypes.complex(EnReal, EnImag), dtype=tf.complex64)

    # calculate spatial spectrum
    spectrum = []
    for i in range(num_samples):
        # establish array steering vector
        if calculateA:
            if ang == 'az':
                a = build_steering_vect_2D(r=sv,
                                           azimuth=tf.cast(angles[0][i], dtype=tf.complex64),
                                           elev=tf.cast(angles[1][num_samples // 2], dtype=tf.complex64),
                                           wavelength=tf.cast(1, dtype=tf.complex64))
            elif ang == 'el':
                a = build_steering_vect_2D(r=sv,
                                           azimuth=tf.cast(angles[0][num_samples // 2], dtype=tf.complex64),
                                           elev=tf.cast(angles[1][i], dtype=tf.complex64),
                                           wavelength=tf.cast(1, dtype=tf.complex64))

        else: a = sv

        H = tf.linalg.matvec((En @ tf.transpose(En, conjugate=True, perm=[0, 2, 1])), a)

        H = tf.reduce_sum(tf.math.multiply(tf.math.conj(a), H), 1)
        spectrum.append(1. / H)

    return tf.transpose(tf.convert_to_tensor(spectrum, dtype=float))


#**********************************#
#   calculate the MUSIC spectrum   #
#**********************************#
def calculate_spectrum_bb(y_pred, f=np.max(fs), r=dist, el=None, sv=None, calculateA=True,
                          slow=None, useSlow=False):
    """
        Calculates the MUSIC spectrum according to P = 1 / (a^H En En^H a).

        @param y_pred -- The estimated noise space vectors.

        @returns -- The estimated spectrum.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]

    EnReal = Lambda(lambda y: y[:, :m])(y_pred)
    EnImag = Lambda(lambda y: y[:, m:])(y_pred)

    En = tf.cast(tf.dtypes.complex(EnReal, EnImag), dtype=tf.complex64)

    # calculate spatial spectrum
    spectrum = []
    for i in range(num_samples):
        # establish array steering vector
        if calculateA:
            # a = tf.cast(ULA_broad(array, axis[i], tf.cast(f, dtype=tf.complex64),
            #                       tf.cast(r, dtype=tf.complex64)), dtype=tf.complex64)
            # a = tf.cast(UCA(array, axis[i]), dtype=tf.complex64)
            # a = tf.cast(build_steering_vect(sv, axis[i]), dtype=tf.complex64)
            if el:
                a = build_steering_vect_2D(r=sv,
                                           azimuth=tf.cast(angles[0][i], dtype=tf.complex64),
                                           elev=tf.cast(el, dtype=tf.complex64),
                                           wavelength=tf.cast(f, dtype=tf.complex64))
            elif useSlow:
                a = tf.cast(build_steering_vect_slow(sv, angles[0][i], slow), dtype=tf.complex64)
            else:
                a = tf.cast(build_steering_vect(sv, angles[0][i]), dtype=tf.complex64)

        else: a = sv

        H = tf.linalg.matvec((En @ tf.transpose(En, conjugate=True, perm=[0, 2, 1])), a)

        H = tf.reduce_sum(tf.math.multiply(tf.math.conj(a), H), 1)
        spectrum.append(1. / H)

    return tf.transpose(tf.convert_to_tensor(spectrum, dtype=float))


#*************************************#
#   calculate the 2d MUSIC spectrum   #
#*************************************#
def calculate_spectrum_bb_2D(y_pred, f=np.max(fs), r=dist, el=None, sv=None, calculateA=True):
    """
        Calculates the MUSIC spectrum according to P = 1 / (a^H En En^H a).

        @param y_pred -- The estimated noise space vectors.

        @returns -- The estimated spectrum.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]

    EnReal = Lambda(lambda y: y[:, :m])(y_pred)
    EnImag = Lambda(lambda y: y[:, m:])(y_pred)

    En = tf.cast(tf.dtypes.complex(EnReal, EnImag), dtype=tf.complex64)

    # calculate spatial spectrum
    spectrum = []
    for i in range(num_samples):
        for j in range(num_samples):
            # establish array steering vector
            if calculateA:
                a = build_steering_vect_2D(r=sv,
                                           azimuth=tf.cast(angles[0][i], dtype=tf.complex64),
                                           elev=tf.cast(angles[1][j], dtype=tf.complex64),
                                           wavelength=tf.cast(f, dtype=tf.complex64))
            else: a = sv

            H = tf.linalg.matvec((En @ tf.transpose(En, conjugate=True, perm=[0, 2, 1])), a)

            H = tf.reduce_sum(tf.math.multiply(tf.math.conj(a), H), 1)
            spectrum.append(1. / H)

    return tf.transpose(tf.convert_to_tensor(spectrum, dtype=float))


#*******************************************#
#   uniform linear array steering vector    #
#*******************************************#
def ULA_broad(array, theta, f, spacing):
    # differentiable version of broadbandMUSIC.py/ULA_broad_action_vector
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.
        @param f -- Holds the frequency of the signals.
        @param spacing -- The distance between the array elements.

        @returns -- The action vector.
    """
    if len(f.shape) > 0:
        As = []
        for elem in array:
            As.append(tf.exp(- 1j * 2 * np.pi * f * spacing * elem * np.sin(theta) / 3e8))
        return tf.stack(As, axis=1)
    else:
        return tf.exp(- 1j * 2 * np.pi * f * spacing * array * np.sin(theta) / 3e8)


#******************************************#
#   uniform linear array steering vector   #
#******************************************#
def UCA(array, theta):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.
        @param f -- Holds the frequency of the signals.
        @param spacing -- The distance between the array elements.

        @returns -- The action vector.
    """
    return tf.exp(- 1j * np.pi * np.sin(theta - 2 * np.pi * array / len(array)))


#********************************************#
#   uniform circular array steering vector   #
#********************************************#
def UCA_broad(array, theta, f, radius):
    """
        Establish the possible mode vectors (steering vectors) given the
        positions of a uniform linear array.

        @param array -- Holds the positions of the array elements.
        @param theta -- The value of the given axis to be evaluated.
        @param f -- Holds the frequency of the signals.
        @param radius -- The radius of the given UCA.

        @returns -- The action vector.
    """
    if len(f.shape) > 0:
        As = []
        for elem in array:
            As.append(tf.exp(- 1j * 2 * np.pi * radius * f *
                      np.sin(theta - 2 * np.pi * elem / len(array)) / 343))
        return tf.stack(As, axis=1)
    else:
        return tf.exp(- 1j * 2 * np.pi * radius * f *
               np.sin(theta - 2 * np.pi * array / len(array)) / 343)


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect(r, azimuth):
    wavelength = 1
    theta = - np.pi / 4
    k = 2 * np.pi / wavelength * np.array([np.sin(theta) * np.cos(azimuth),
                                           np.sin(theta) * np.sin(azimuth), np.cos(theta)])

    return tf.exp(-1j * tf.cast(tf.tensordot(r, tf.cast(k, dtype=tf.float32), axes=1),
                                dtype=tf.complex64))



#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect_slow(r, azimuth, slow):
    wavelength = 1
    theta = - np.pi / 4
    k = 2 * np.pi / wavelength * np.array([np.sin(theta) * np.cos(azimuth),
                                           np.sin(theta) * np.sin(azimuth), np.cos(theta)])

    return tf.exp(-1j * tf.cast(slow, dtype=tf.complex64) *
                  tf.cast(tf.tensordot(r, tf.cast(k, dtype=tf.float32), axes=1), dtype=tf.complex64))


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect_2D(r, azimuth, elev, wavelength):
    k = [tf.sin(elev) * tf.cos(azimuth), tf.sin(elev) * tf.sin(azimuth), tf.cos(elev)]
    k = tf.tensordot(2 * np.pi / (wavelength + tf.keras.backend.epsilon()), k, axes=0)

    return tf.exp(-1j * tf.reduce_sum(tf.cast(r, dtype=tf.complex64) * k, axis=-1))


#************************************************#
#   build steering vectors from r and azimuth    #
#************************************************#
def build_steering_vect_2D_test(r, azimuth, elev, wavelength):
    k = [tf.sin(elev) * tf.cos(azimuth), tf.sin(elev) * tf.sin(azimuth), tf.cos(elev)]
    k = tf.transpose(2 * np.pi / (wavelength + tf.keras.backend.epsilon()) * k)

    r = tf.transpose(r, perm=[1, 0, 2])

    return tf.transpose(tf.exp(-1j * tf.reduce_sum(tf.cast(r, dtype=tf.complex64) * k, axis=-1)))


#**********************************#
#   calculate the MUSIC spectrum   #
#**********************************#
def calculate_spectrum(y_pred):
    """
        Calculates the MUSIC spectrum according to P = 1 / (a^H En En^H a).

        @param y_pred -- The estimated noise space vectors.

        @returns -- The estimated spectrum.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]

    EnReal = Lambda(lambda y: y[:, :m])(y_pred)
    EnImag = Lambda(lambda y: y[:, m:])(y_pred)

    En = tf.cast(tf.dtypes.complex(EnReal, EnImag), dtype=tf.complex64)
    # calculate spatial spectrum
    spectrum = []
    for axis in angles:
        for i in range(num_samples):
            # establish array steering vector
            a = tf.cast(ULA_action_vector(array, axis[i]), dtype=tf.complex64)
            # a = np.repeat(a[np.newaxis, :], batch_size, axis=0)

            H = tf.linalg.matvec((En @ tf.transpose(En, conjugate=True, perm=[0, 2, 1])), a)

            H = tf.reduce_sum(tf.math.multiply(tf.math.conj(a), H), 1)
            spectrum.append(1. / H)

    return tf.transpose(tf.convert_to_tensor(spectrum, dtype=float))


#****************************#
#   calculate permutations   #
#****************************#
def permutations(predDoA):
    """
        Calculates all permutations of the given list.

        @param predDoA -- The estimated DoA angles to be permuted.

        @returns -- All permutations of the estimated DoA.
    """
    if len(predDoA) == 0:
        return []
    if len(predDoA) == 1:
        return [predDoA]

    perms = []
    for i in range(len(predDoA)):
       remaining = predDoA[:i] + predDoA[i + 1:]

       for perm in permutations(remaining):
           perms.append([predDoA[i]] + perm)

    return perms