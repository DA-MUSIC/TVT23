####################################################################################################
#                                             models.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 17/04/21                                                                                #
#                                                                                                  #
# Purpose: Definition of the architecture of the augmentation for the MUSIC algorithm.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import Regularizer, L1, L1L2
from tensorflow.keras.utils import plot_model

from scipy import linalg
from scipy import signal

from syntheticEx import *
from bbSynthEx import fs, dist, snapshots
from utils import *


#********************#
#   initialization   #
#********************#
n = m - d   # number of noise vectors
r = angles.shape[1]   # resolution (i.e. angle grid size)


#***********#
#   model   #
#***********#
def deep_aug_MUSIC():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)

    # create covariance from measurements
    y = GRU(2 * m)(y)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_sca(scale, f):
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)

    # create covariance from measurements
    y = GRU(2 * m * scale)(y)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_real():
    x = Input((2 * m, snapshots))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    # estimate frequencies of signals
    f = GRU(2 * m)(s)
    f = Dense(2, activation='sigmoid')(f)

    f_i = Lambda(lambda y: y[:, 0])(f)
    d_i = Lambda(lambda y: y[:, 1])(f)

    # scale appropriately
    f_i = tf.math.scalar_mul(scalar=float(len(fs)), x=f_i)
    d_i = tf.math.scalar_mul(scalar=3430 / float(len(fs)), x=d_i)

    y = Lambda(lambda y: calculate_spectrum_bb(y[0], f=y[1], r=y[2]))([y, f_i, d_i])

    # y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_isv():
    x = Input((2 * m, snapshots))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    # estimate frequencies of signals
    a = GRU(2 * m)(s)
    a = Dense(2 * m, activation='sigmoid')(a)

    # transform to complex numbers
    aReal = Lambda(lambda y: y[:, :m])(a)
    aImag = Lambda(lambda y: y[:, m:])(a)
    a = tf.complex(aReal, aImag)

    y = Lambda(lambda y: calculate_spectrum_bb(y[0], sv=y[1], calculateA=False))([y, a])

    # y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_stv():
    x = Input((2 * m, snapshots))
    sv = Input((m, 3))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    # # estimate frequencies of signals
    # a = GRU(2 * m)(s)
    # a = Dense(2 * m, activation='sigmoid')(a)
    #
    # # transform to complex numbers
    # aReal = Lambda(lambda y: y[:, :m])(a)
    # aImag = Lambda(lambda y: y[:, m:])(a)
    # a = tf.complex(aReal, aImag)

    y = Lambda(lambda y: calculate_spectrum_bb(y[0], sv=y[1], calculateA=True))([y, sv])

    # y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = BatchNormalization()(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return [x, sv], y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_stv_noImag():
    x = Input((m, snapshots))
    sv = Input((m, 3))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    # # estimate frequencies of signals
    # a = GRU(2 * m)(s)
    # a = Dense(2 * m, activation='sigmoid')(a)
    #
    # # transform to complex numbers
    # aReal = Lambda(lambda y: y[:, :m])(a)
    # aImag = Lambda(lambda y: y[:, m:])(a)
    # a = tf.complex(aReal, aImag)

    y = Lambda(lambda y: calculate_spectrum_bb(y[0], sv=y[1], calculateA=True))([y, sv])

    # y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = BatchNormalization()(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return [x, sv], y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_stv_slo_noImag():
    x = Input((m, snapshots))
    sv = Input((m, 3))
    slo = Input((1))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(m)(s)
    y = Dropout(0.3)(y)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum_bb(y[0], sv=y[1], calculateA=True,
                                               slow=y[2], useSlow=True))([y, sv, slo])

    y = BatchNormalization()(y)

    y = Dense(m, activation = 'relu')(y)
    y = Dropout(0.3)(y)
    y = Dense(m, activation = 'relu')(y)
    y = Dropout(0.3)(y)
    y = Dense(m, activation = 'relu')(y)
    y = Dropout(0.3)(y)

    y = Dense(d)(y)

    return [x, sv, slo], y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_stv_2D_noImag():
    x = Input((m, snapshots))
    sv = Input((m, 3))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum_bb_2D(y[0], sv=y[1], calculateA=True))([y, sv])

    # y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = BatchNormalization()(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return [x, sv], y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_2D_estF():
    x = Input((2 * m, snapshots))
    sv = Input((m, 3))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    # estimate frequencies of signals
    f = GRU(2 * m)(s)
    f = Dense(2, activation='sigmoid')(f)

    f_i = Lambda(lambda y: y[:, 0])(f)
    e_i = Lambda(lambda y: y[:, 1])(f)

    # scale appropriately
    f_i = tf.math.scalar_mul(scalar=1000., x=f_i)
    e_i = tf.math.scalar_mul(scalar=2 * np.pi, x=e_i)

    y = Lambda(lambda y: calculate_spectrum_bb(y[0], sv=y[1], f=y[2], el=y[3]))([y, sv, f_i, e_i])

    # y = Lambda(lambda y: calculate_spectrum_bb(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return [x, sv], y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_2D_con():
    x = Input((2 * m, snapshots))
    sv = Input((m, 3))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y_az = Lambda(lambda y: calculate_spectrum_ang(y[0], ang='az', sv=y[1], calculateA=True))([y, sv])
    y_el = Lambda(lambda y: calculate_spectrum_ang(y[0], ang='el', sv=y[1], calculateA=True))([y, sv])

    # y = Concatenate()(specs)
    y = tf.stack([y_az, y_el], axis=1)

    y = BatchNormalization()(y)

    y = Conv1D(filters=180, strides=2, kernel_size=2)(y)

    y = Flatten()(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return [x, sv], y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_bb():
    x = Input((2 * m, snapshots))

    g = Permute((2, 1))(x)
    g = BatchNormalization()(g)

    specs = []
    bins = 10
    for i in range(bins):
        # create covariance from measurements
        y = (GRU(2 * m)(g))
        y = Dense(2 * m * m)(y)
        y = Reshape((2 * m, m))(y)

        # transform to complex numbers
        yReal = Lambda(lambda y: y[:, :m])(y)
        yImag = Lambda(lambda y: y[:, m:])(y)
        y = tf.complex(yReal, yImag)

        # eigenvector decomposition
        yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

        # chose n smallest eignevalues/eigenvectors
        yVec = Lambda(lambda y: y[:, :, d:])(yVec)

        # transform back to real and imag part stacked
        yReal = tf.math.real(yVec)
        yImag = tf.math.imag(yVec)
        y = Concatenate(axis=1)([yReal, yImag])

        y = Lambda(lambda y: calculate_spectrum_bb(y, fs[(i+1) * len(fs)//bins - 1]))(y)

        specs.append(y)

    # y = Concatenate()(specs)
    y = tf.stack(specs, axis=1)

    y = Flatten()(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_fil():
    x = Input((2 * m, 10, snapshots))

    g = Permute((2, 1, 3))(x)

    specs = []
    bins = 10
    for i in range(bins):
        y = Lambda(lambda y: y[:, i])(g)
        y = Permute((2, 1))(y)
        y = BatchNormalization()(y)

        # create covariance from measurements
        y = (GRU(2 * m)(y))
        y = Dense(2 * m * m)(y)
        y = Reshape((2 * m, m))(y)

        # transform to complex numbers
        yReal = Lambda(lambda y: y[:, :m])(y)
        yImag = Lambda(lambda y: y[:, m:])(y)
        y = tf.complex(yReal, yImag)

        # eigenvector decomposition
        yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

        # chose n smallest eignevalues/eigenvectors
        yVec = Lambda(lambda y: y[:, :, d:])(yVec)

        # transform back to real and imag part stacked
        yReal = tf.math.real(yVec)
        yImag = tf.math.imag(yVec)
        y = Concatenate(axis=1)([yReal, yImag])

        y = Lambda(lambda y: calculate_spectrum_bb(y, fs[(i+1) * len(fs)//bins - 1]))(y)

        y = Dense(2 * m, activation='relu')(y)
        y = Dense(2 * m, activation='relu')(y)
        y = Dense(2 * m, activation='relu')(y)

        specs.append(y)

    # y = Concatenate()(specs)
    y = tf.stack(specs, axis=1)

    y = Flatten()(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_est_f():
    x = Input((2 * m, snapshots))

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    # create covariance from measurements
    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # chose n smallest eignevalues/eigenvectors
    yVec = Lambda(lambda y: y[:, :, d:])(yVec)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    # estimate frequencies of signals
    f = GRU(2 * m)(s)
    f = Dense(d, activation='sigmoid')(f)
    f = tf.math.scalar_mul(scalar=float(len(fs)), x=f)

    # estimate DoA by computing spectra for each frequency
    thetas = []
    for i in range(d):
        f_i = Lambda(lambda y: y[:, i])(f)

        the = Lambda(lambda y: calculate_spectrum_bb(y[0], f=y[1]))([y, f_i])

        the = Dense(2 * m, activation='relu')(the)
        the = Dense(2 * m, activation='relu')(the)
        the = Dense(2 * m, activation='relu')(the)

        the = Dense(1)(the)
        thetas.append(the)

    y = Concatenate()(thetas)

    return x, y


#***********#
#   model   #
#***********#
def deep_aug_MUSIC_bb_est_f():
    x = Input((2 * m, snapshots))

    s = BatchNormalization()(x)
    s = Permute((2, 1))(s)
    s = BatchNormalization()(s)

    # estimate frequencies of signals
    f = GRU(2 * m)(s)
    f = Dense(d, activation='sigmoid')(f)
    f = tf.math.scalar_mul(scalar=float(len(fs)), x=f)

    thetas = []
    for i in range(d):
        # create covariance from measurements
        y = (GRU(2 * m)(s))
        y = Dense(2 * m * m)(y)
        y = Reshape((2 * m, m))(y)

        # transform to complex numbers
        yReal = Lambda(lambda y: y[:, :m])(y)
        yImag = Lambda(lambda y: y[:, m:])(y)
        y = tf.complex(yReal, yImag)

        # eigenvector decomposition
        yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

        # chose n smallest eignevalues/eigenvectors
        yVec = Lambda(lambda y: y[:, :, d:])(yVec)

        # transform back to real and imag part stacked
        yReal = tf.math.real(yVec)
        yImag = tf.math.imag(yVec)
        y = Concatenate(axis=1)([yReal, yImag])

        f_i = Lambda(lambda y: y[:, i])(f)

        y = Lambda(lambda y: calculate_spectrum_bb(y[0], f=y[1]))([y, f_i])

        the = Dense(2 * m, activation='relu')(y)
        the = Dense(2 * m, activation='relu')(the)
        the = Dense(2 * m, activation='relu')(the)

        the = Dense(1)(the)
        thetas.append(the)

    y = Concatenate()(thetas)

    return x, y


#***********#
#   model   #
#***********#
def aug_E2E_alternative():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Dense(angles.shape[1])(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(d)(y)

    return x, y


#***********#
#   model   #
#***********#
def deepMUSIC():
    q = 24

    inp = Input((3, m, m))

    x = BatchNormalization()(inp)
    x = Permute((2, 3, 1))(x)

    out = []
    for i in range(q):
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3))(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Flatten()(y)

        y = Dense(1024)(y)

        # y = Dropout(0.3)(y)

        y = Activation('softmax')(y)

        y = Dense(r//q)(y)

        out.append(y)

    return inp, out


#***********#
#   model   #
#***********#
def aug_MUSIC_est_d():
    x = Input((2 * m, snapshots), batch_size=16)

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)

    # create covariance from measurements
    y = GRU(2 * m)(y)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    yVal = tf.stop_gradient(yVal)

    # transform eigenvalues to real and imag part stacked
    yReal = tf.math.real(yVal)
    yImag = tf.math.imag(yVal)
    yVal = Concatenate(axis=1)([yReal, yImag])

    # probability p_i of choosing eigenvector v_i for i in {1, ..., m}
    y = Dense(2 * m, activation='relu')(yVal)
    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)
    p_i = Dense(m, activation='sigmoid')(y)

    # p_i = Lambda(lambda y: tf.math.scalar_mul(m, y))(p_i)

    p_i = Lambda(lambda y: (y - (tf.stop_gradient(y) - tf.round(y))))(p_i)

    # diag. matrix with p_i
    P = Lambda(lambda y: tf.cast(tf.linalg.diag(y), dtype=tf.complex64))(p_i)

    # P = tf.stop_gradient(P)

    # En = P * V
    yVec = Lambda(lambda y: tf.matmul(y[0], y[1]))([P, yVec])

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(m - 1)(y)

    # y = Concatenate()([y, p_i])

    return x, [y, p_i]


#***********#
#   model   #
#***********#
def aug_MUSIC_est_d_no_evd():
    x = Input((2 * m, snapshots), batch_size=16)

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Dense(2 * m * m, activation='relu')(y)
    y = Dense(2 * m * m, activation='relu')(y)

    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    yVec = tf.complex(yReal, yImag)

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(m - 1)(y)

    return x, y


#***********#
#   model   #
#***********#
def aug_MUSIC_soft_q():
    x = Input((2 * m, snapshots), batch_size=16)

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # transform eigenvalues to real and imag part stacked
    yReal = tf.math.real(yVal)
    yImag = tf.math.imag(yVal)
    yVal = Concatenate(axis=1)([yReal, yImag])

    # probability p_i of choosing eigenvector v_i for i in {1, ..., m}
    y = Dense(2 * m, activation='relu')(yVal)
    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)
    p_i = Dense(m, activation='sigmoid')(y)
    # p_i = Dense(m, activation='softmax')(y)
    #
    # p_i = Lambda(lambda y: tf.math.scalar_mul(m, y))(p_i)

    # diag. matrix with p_i
    P = Lambda(lambda y: tf.cast(tf.linalg.diag(y), dtype=tf.complex64))(p_i)

    # En = P * V
    yVec = Lambda(lambda y: tf.matmul(y[0], y[1]))([P, yVec])

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(m - 1)(y)

    return x, y


#***********#
#   model   #
#***********#
def aug_MUSIC_soft_q_est():
    x = Input((2 * m, snapshots), batch_size=16)

    s = Permute((2, 1))(x)
    s = BatchNormalization()(s)

    y = GRU(2 * m)(s)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # transform eigenvalues to real and imag part stacked
    yReal = tf.math.real(yVal)
    yImag = tf.math.imag(yVal)
    yVal = Concatenate(axis=1)([yReal, yImag])

    yVal = tf.stop_gradient(yVal)

    # estimate number of sources from eigenvalues
    num = Dense(2 * m * m, activation='relu')(yVal)
    num = Dense(2 * m * m, activation='relu')(num)
    num = Dense(2 * m * m, activation='relu')(num)
    num = Dense(4, 'softmax')(num)

    # probability p_i of choosing eigenvector v_i for i in {1, ..., m}
    y = Dense(2 * m, activation='relu')(yVal)
    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)
    p_i = Dense(m, activation='sigmoid')(y)

    # p_i = Dense(m, activation='softmax')(y)
    #
    # p_i = Lambda(lambda y: tf.math.scalar_mul(m, y))(p_i)

    # diag. matrix with p_i
    P = Lambda(lambda y: tf.cast(tf.linalg.diag(y), dtype=tf.complex64))(p_i)

    # # !!! En = P * V
    yVec = Lambda(lambda y: tf.matmul(y[0], y[1]))([P, yVec])

    # # En = V * P
    # yVec = Lambda(lambda y: tf.matmul(y[0], y[1]))([yVec, P])

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(m - 1)(y)

    return x, [y, num]


#***********#
#   model   #
#***********#
def aug_MUSIC_hard_q():
    x = Input((2 * m, snapshots), batch_size=16)

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # transform eigenvalues to real and imag part stacked
    yReal = tf.math.real(yVal)
    yImag = tf.math.imag(yVal)
    yVal = Concatenate(axis=1)([yReal, yImag])

    yVal = tf.stop_gradient(yVal)

    # probability p_i of choosing eigenvector v_i for i in {1, ..., m}
    y = Dense(2 * m, activation='relu')(yVal)
    y = Dense(2 * m, activation='relu')(y)
    y = Dense(2 * m, activation='relu')(y)
    p_i = Dense(m, activation='sigmoid')(y)

    p_i = Lambda(lambda y: (y - (tf.stop_gradient(y) - tf.round(y))))(p_i)

    # diag. matrix with p_i
    P = Lambda(lambda y: tf.cast(tf.linalg.diag(y), dtype=tf.complex64))(p_i)

    # # En = P * V
    # yVec = Lambda(lambda y: tf.matmul(y[0], y[1]))([P, yVec])

    # En = V * P
    yVec = Lambda(lambda y: tf.matmul(y[0], y[1]))([yVec, P])

    # transform back to real and imag part stacked
    yReal = tf.math.real(yVec)
    yImag = tf.math.imag(yVec)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum(y))(y)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(m - 1)(y)

    return x, y


#***********#
#   model   #
#***********#
def aug_MUSIC_est_d_MB_NN():
    x = Input((2 * m, snapshots), batch_size=1)

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)

    # create covariance from measurements
    y = GRU(2 * m)(y)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # # get clustering threshold
    # thr = Dense(2 * m, activation = 'relu')(yVal)
    # thr = Dense(2 * m, activation = 'relu')(thr)
    # thr = Dense(2 * m, activation = 'relu')(thr)
    # thr = Dense(1, activation = 'sigmoid')(thr)

    # chose n smallest eignevalues/eigenvectors
    specs = []
    for i, lam in enumerate(yVal):
        # t = Lambda(lambda y: y[i])(thr)
        t = 0.4

        # estimate multiplicity of smallest eigenvalue...
        mask = Lambda(lambda y: tf.greater(tf.add(abs(y[0][-1]), y[1]), abs(y[0])))([lam, t])

        # and select eigenvectors accordingly
        v = Lambda(lambda y: y[i, :, :])(yVec)
        En = Lambda(lambda y: tf.boolean_mask(y[0], y[1]))([v, mask])
        En = tf.expand_dims(En, axis=0)

        # transform back to real and imag part stacked
        yReal = tf.math.real(En)
        yImag = tf.math.imag(En)
        En_satck = Concatenate(axis=1)([yReal, yImag])

        specs.append(Lambda(lambda y: calculate_spectrum(y))(En_satck))

    y = tf.concat(specs, axis=0)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(m - 1)(y)

    return x, y


#***********#
#   model   #
#***********#
def aug_MUSIC_est_d_MB():
    x = Input((2 * m, snapshots))

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)

    # create covariance from measurements
    y = GRU(2 * m)(y)
    y = Dense(2 * m * m)(y)
    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eigh(y))(y)

    t = 0.4

    # estimate multiplicity of smallest eigenvalue...
    mask = Lambda(lambda y: tf.greater(tf.add(abs(y[0][-1]), y[1]), abs(y[0])))([yVal, t])
    mask = tf.repeat(tf.expand_dims(mask, axis=1), m, axis=1)

    # and select eigenvectors accordingly
    En = Lambda(lambda y: tf.where(y[1], y[0], tf.zeros_like(y[0])))([yVec, mask])

    # transform back to real and imag part stacked
    yReal = tf.math.real(En)
    yImag = tf.math.imag(En)
    En_satck = Concatenate(axis=1)([yReal, yImag])

    y = Lambda(lambda y: calculate_spectrum(y))(En_satck)

    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)
    y = Dense(2 * m, activation = 'relu')(y)

    y = Dense(m - 1)(y)

    return x, y


#***********#
#   model   #
#***********#
def aug_MUSIC_est_d_MB_switch():
    x = Input((2 * m, snapshots), batch_size=16)

    y = Permute((2, 1))(x)
    y = BatchNormalization()(y)
    y = GRU(2 * m)(y)

    y = Dense(2 * m * m)(y)

    y = Reshape((2 * m, m))(y)

    # transform to complex numbers
    yReal = Lambda(lambda y: y[:, :m])(y)
    yImag = Lambda(lambda y: y[:, m:])(y)
    y = tf.complex(yReal, yImag)

    # eigenvector decomposition
    yVal, yVec = Lambda(lambda y: tf.linalg.eig(y))(y)

    # switch
    out = []
    for d in range(2, 6):

        # chose n smallest eignevalues/eigenvectors
        y = Lambda(lambda y: y[:, :, d:])(yVec)

        # transform back to real and imag part stacked
        yReal = tf.math.real(y)
        yImag = tf.math.imag(y)
        y = Concatenate(axis=1)([yReal, yImag])

        y = Lambda(lambda y: calculate_spectrum(y))(y)

        y = Dense(2 * m, activation = 'relu')(y)
        y = Dense(2 * m, activation = 'relu')(y)
        y = Dense(2 * m, activation = 'relu')(y)

        out.append(Dense(d)(y))

    # y = Concatenate()(out)

    return x, out


#***********#
#   model   #
#***********#
def separate_est_d(aug):
    x = Input((2 * m, snapshots), batch_size=16)

    inp = aug.input
    outputs = [layer.output for layer in aug.layers]

    # select output layer forming the spectrum and final layer (with DoA)
    functor = [K.function([inp], [outputs[9]]), K.function([inp], [outputs[-2]])]

    lambdas, DoA = [f([x]) for f in functor]

    y = lambdas[0][0][0]

    # transform back to real and imag part stacked
    yReal = tf.math.real(y)
    yImag = tf.math.imag(y)
    y = Concatenate(axis=1)([yReal, yImag])

    y = Dense(2 * m * m, activation='relu')(y)
    y = Dense(2 * m * m, activation='relu')(y)
    y = Dense(2 * m * m, activation='relu')(y)

    y = Dense(4, activation='softmax')(y)

    return x, y


#***********#
#   model   #
#***********#
def soft_q_est_d_NON_TRAIN(aug, clas):
    x = Input((2 * m, snapshots), batch_size=16)

    inp = aug.input
    outputs = [layer.output for layer in aug.layers]

    # select output layers
    functor = [K.function([inp], [outputs[9]]), K.function([inp], [outputs[-2]])]

    lambdas, DoA = [f([x]) for f in functor]

    inp = clas.input
    outputs = [layer.output for layer in clas.layers]

    # select output layers
    functor = [K.function([inp], [outputs[-1]])]

    y = [f([x]) for f in functor]

    return x, [DoA[0], y[0]]