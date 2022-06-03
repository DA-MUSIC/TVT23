####################################################################################################
#                                            losses.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 25/03/21                                                                                #
#                                                                                                  #
# Purpose: Definitions of custom losses used to train a neural augmentation of the MUSIC           #
#          algorithm.                                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import Regularizer

from scipy import signal
from scipy.stats import laplace

from syntheticEx import angles
from utils import *


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)
tf.random.set_seed(42)


#************************#
#   MUSIC spectrum mse   #
#************************#
def mseSpectrum(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The "perfect" spectrum.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The mse of the estimated spectrum and the spectrum given by y_true.
    """
    y = calculate_spectrum(y_pred)

    y = LayerNormalization()(y)
    y_true = LayerNormalization()(y_true)

    return K.mean(K.square(y - y_true), axis=-1)


#****************************************#
#   inverse spectrum at location peaks   #
#****************************************#
def inversePeaks(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The true DoA of the sources.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The inverse of the sum of the estimated spectrum at the true DoA
                    (thereby favouring large values at these locations).
    """
    y = calculate_spectrum(y_pred)

    # normalize spectrum to [-1, 1] and shift +1 up to assure non-negative values
    # ! error occurred ! -> possibly float rounding error -> shift up > 1 to be sure
    shift_up = 2
    y = LayerNormalization()(y) + shift_up

    peaks = tf.gather(y, indices=tf.cast(y_true, dtype='int32'), axis=1, batch_dims=1)

    # sum up all inverse peaks and normalize
    inverse_peaks =  shift_up * tf.reduce_sum(1. / peaks, axis=-1) / y_true.shape[1]

    return inverse_peaks # + tf.reduce_sum(y, axis=-1) / num_samples


#***************************************************#
#   difference between peaks and rest of spectrum   #
#***************************************************#
def PeakSpektrumDiff(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.
        Caution! - This loss becomes negative instead of smaller!

        @param y_true -- The true DoA of the sources.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The difference between the peaks and the mean of the rest
                    of the spectrum.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]

    y = calculate_spectrum(y_pred)

    # normalize spectrum to [-1, 1] and shift +1 up to assure non-negative values
    # ! error occurred ! -> possibly float rounding error -> shift up > 1 to be sure
    shift_up = 2
    y = LayerNormalization()(y) + shift_up

    peaks = tf.gather(y, indices=tf.cast(y_true, dtype='int32'), axis=1, batch_dims=1)

    # create spectrum without peaks
    for a in range(y_true.shape[1]):
        y_remove = tf.sparse.SparseTensor(indices=[[j, y_true[j, a]] for j in range(batch_size)],
                                          values=peaks[:, a],
                                          dense_shape=[batch_size, num_samples])
        y = y - tf.sparse.to_dense(y_remove, default_value=0.)

    # return difference between the mean of the rest of the spectrum and the peaks
    return tf.reduce_sum(tf.math.subtract(tf.reduce_mean(y), peaks), axis=-1)


#***********************************************#
#   mse of the estimated DoA and the true DoA   #
#***********************************************#
def mseDoA(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The true DoA of the sources.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The (angular) mse of the estimated DoA and the true DoA.
    """
    batch_size = y_pred.shape[0]
    num_samples = angles.shape[1]
    d = y_true.shape[1]

    y = calculate_spectrum(y_pred)
    y = LayerNormalization()(y) + 1   # normalize to [0, 2]

    beta = 1e3
    y_range = tf.range(y.shape.as_list()[-1], dtype=y.dtype)
    DoAs = []
    for _ in range(d):

        # soft-argmax
        DoA = tf.reduce_sum(tf.nn.softmax(y * beta) * y_range, axis=-1)

        DDoA = tf.expand_dims(tf.dtypes.cast(DoA, tf.int64), axis=1)
        DDoA = tf.dtypes.cast(DDoA, float)

        DoAs.append(DoA)

        if d > 1:
            # get distances to true DoA
            dist = tf.sort(abs(((y_true - DDoA) + num_samples/2) % num_samples - num_samples/2))

            # build range around estimated DoA using second smallest distance
            indices = []
            for b in range(batch_size):
                for idx in range(DDoA[b] - dist[b, 1]//2, DDoA[b] + dist[b, 1]//2 + 1):
                    indices.append([b, idx % num_samples])

            # remove maximal value and range around it
            values = tf.gather_nd(y, indices=tf.convert_to_tensor(indices, dtype='int32'))
            y_remove = tf.sparse.SparseTensor(indices=indices, values=values,
                                              dense_shape=[batch_size, num_samples])
            y = y - tf.sparse.to_dense(tf.sparse.reorder(y_remove), default_value =0.)

    DoAs = tf.transpose(tf.convert_to_tensor(DoAs, dtype=float))

    # account for angular overflow (pi / 2 = - pi /2)
    diff = ((tf.sort(DoAs) - tf.sort(y_true)) + num_samples/2) % num_samples - num_samples/2

    return K.mean(diff ** 2, axis=-1) / ((num_samples/4) ** 2)


#***************#
#  mse of evd   #
#***************#
def evd(y_true, y_pred):
    """
        Defines a custom loss to estimate performance of subspace estimator.

        @param y_true -- The evd of the measurements.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The mse of the estimated evd and the true evd given by y_true.
    """

    return K.mean(K.mean(tf.math.subtract(y_pred, y_true) ** 2, axis=-1), axis=-1)


#******************#
#   angular rmse   #
#******************#
def angular_rmse(y_true, y_pred):
    """
        Defines a custom loss for a DoA estimator.

        @param y_true -- The evd of the measurements.
        @param y_pred -- The estimated noise space vectors.

        @returns -- The rmse with respect to the angles.
    """
    # angular difference
    diff = tf.math.floormod((tf.subtract(tf.sort(y_pred), tf.sort(y_true)) + np.pi / 2), np.pi) - np.pi / 2
    return tf.reduce_mean((diff ** 2), axis=-1) ** (1 / 2)


#***********************************#
#   mean minimal permutation rmse   #
#***********************************#
def perm_rmse(predDoA, trueDoA):
    # differentiable version of errorMeasures.py/mean_min_perm_rmse
    """
        Calculates the mean of all the samples of the minimal rmse of
        (all permutations of) the predicted DoA and the true DoA.

        @param predDoA -- The estimated DoA angles in radians (same length as true DoA).
        @param trueDoA -- The ground truth DoA angles in radians.

        @returns -- The mean of minimal rmse.
    """
    num_samples = trueDoA.shape[0]
    num_sources = trueDoA.shape[1]

    # get permutations of estimated DoA
    allPerms = np.zeros((num_samples, np.math.factorial(num_sources), num_sources))
    for i in range(num_samples):
        allPerms[i] = permutations(list(predDoA[i]))

    # angular difference
    diff = tf.math.floormod(((Subtract()([allPerms, trueDoA])) + angles[0, -1] / 2),
                            (angles[0, -1] - angles[0, 0])) - angles[0, -1] / 2

    # rmse
    diff = tf.reduce_mean((diff ** 2), axis=-1) ** (1 / 2)

    # return minimal rmse as error
    return tf.math.reduce_min(diff, axis=-1)


#***********************************#
#   mean minimal permutation rmse   #
#***********************************#
def perm_rmse_non_angular(predDoA, trueDoA):
    """
        Calculates the mean of all the samples of the minimal rmse of
        (all permutations of) the predicted DoA and the true DoA.

        @param predDoA -- The estimated DoA angles in radians (same length as true DoA).
        @param trueDoA -- The ground truth DoA angles in radians.

        @returns -- The mean of minimal rmse.
    """
    num_samples = trueDoA.shape[0]
    num_sources = trueDoA.shape[1]

    # get permutations of estimated DoA
    allPerms = np.zeros((num_samples, np.math.factorial(num_sources), num_sources))
    for i in range(num_samples):
        allPerms[i] = permutations(list(predDoA[i]))

    # difference
    diff = Subtract()([allPerms, trueDoA])

    # rmse
    diff = tf.reduce_mean((diff ** 2), axis=-1) ** (1 / 2)

    # return minimal rmse as error
    return tf.math.reduce_min(diff, axis=-1)


#***************#
#   mean rmse   #
#***************#
def rmse_est_d(trueDoA, predDoA):

    p_i = predDoA[:, m - 1:]
    predDoA = predDoA[:, :m - 1]

    num_true = tf.math.argmax(trueDoA, axis=1)

    diff = []
    for i, elem in enumerate(num_true):
        rmspe = perm_rmse(tf.expand_dims(trueDoA[i, :elem], axis=0),
                          tf.expand_dims(predDoA[i, :elem], axis=0))

        # diff.append(rmspe + tf.reduce_sum(p_i[i, :elem]) + tf.reduce_sum((1 - p_i[i, elem:])))
        # diff.append(rmspe + tf.reduce_sum([p * tf.abs(1 - p) for p in p_i[i]]))
        diff.append(rmspe)

    return tf.reduce_mean(diff)


#***************#
#   mean rmse   #
#***************#
def rmse_est_d_single(trueDoA, predDoA):

    num_true = tf.math.argmax(trueDoA, axis=1)

    diff = []
    for i, elem in enumerate(num_true):
        rmspe = perm_rmse(tf.expand_dims(predDoA[i, :elem], axis=0),
                          tf.expand_dims(trueDoA[i, :elem], axis=0))

        diff.append(rmspe)

    return tf.reduce_mean(diff)


#***************#
#   mean rmse   #
#***************#
def mse_est_d_p_i(trueDoA, predDoA):

    num = 8 - tf.reduce_sum(predDoA, axis=-1)
    num_true = tf.math.argmax(trueDoA, axis=1)

    return tf.keras.metrics.mean_squared_error(num, tf.cast(num_true, dtype=tf.float32))


#***************#
#   mean rmse   #
#***************#
def rmse_est_d_switch(trueDoA, predDoA):

    num_true = tf.math.argmax(trueDoA, axis=1)

    diff = []
    for i, elem in enumerate(num_true):
        ind = ((elem - 1) * elem // 2) - 1   # gives index 0, 2, 5 for 2, 2, 3, 3, 3, 4, 4, 4, 4

        rmspe = perm_rmse(tf.expand_dims(trueDoA[i, :elem], axis=0),
                          tf.expand_dims(predDoA[i, ind: ind + elem], axis=0))

        diff.append(rmspe)

    return tf.reduce_mean(diff)


#***************#
#   mean rmse   #
#***************#
def rmse_est_d_switch_separate(trueDoA, predDoA):

    num_true = tf.math.argmax(trueDoA, axis=1)

    diff = []
    for i, elem in enumerate(num_true):

        if num_true[i] != predDoA.shape[1]: diff.append([0])

        else:
            rmspe = perm_rmse(tf.expand_dims(trueDoA[i, :elem], axis=0),
                              tf.expand_dims(predDoA[i, :], axis=0))

            diff.append(rmspe)

    return tf.reduce_mean(diff)


#**********************#
#   categorical loss   #
#**********************#
def cat_est_d(trueDoA, predDoA):

    num_true = tf.math.argmax(trueDoA, axis=1) - 2

    print(num_true)

    return tf.keras.losses.SparseCategoricalCrossentropy()(num_true, predDoA)


#**********************************#
#   mean absolute periodic error   #
#**********************************#
def mape(predDoA, trueDoA):
    """
        Calculates the mean absolute periodic error (MAE with periodicity).
        !!! only used for one SINGLE source - no periodicity...

        @param predDoA -- The estimated DoA angles in radians (same length as true DoA).
        @param trueDoA -- The ground truth DoA angles in radians.

        @returns -- The mean of minimal rmse.
    """
    num_samples = trueDoA.shape[0]
    num_sources = trueDoA.shape[1]

    # angular difference
    diff = tf.math.floormod(((Subtract()([predDoA, trueDoA])) + angles[0, -1] / 2),
                            (angles[0, -1] - angles[0, 0])) - angles[0, -1] / 2

    # return mean mape
    return tf.math.reduce_mean(tf.abs(diff), axis=-1)