####################################################################################################
#                                         regularizer.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 24/06/21                                                                                #
#                                                                                                  #
# Purpose: Definitions of custom layers used to train a neural augmentation of the MUSIC           #
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

from scipy.stats import laplace
from syntheticEx import *


#*******************#
#   orthogonality   #
#*******************#
class OrthRegularizer(Regularizer):
    """
        Defines custom  (activity) regularizer by calculating the dot product of the noise
        subspace to estimate performance of subspace estimator.
    """

    def __init__(self, d=5, m=32, w1=1, w2=1):
        """
            @param d -- The number of sources.
            @param m -- The number of array elements.
            @param w1 -- The weighting of the regularization, y1^H y2 != 0.
            @param w2 -- The weighting of the regularization, y1^H y1 != 1.
        """
        self.d = d
        self.m = m
        self.w1 = w1
        self.w2 = w2

    def __call__(self, En_stack):
        """
            @param En_stack-- The stacked real and imaginary part of the noise subspace.

            @returns -- The summed up dot product of all vectors.
        """
        yReal = Lambda(lambda y: y[:, :32])(En_stack)
        yImag = Lambda(lambda y: y[:, 32:])(En_stack)

        yRealT = Permute((2, 1))(yReal)
        yImagT = Permute((2, 1))(yImag)

        # dot product, i.e. y^H y
        y1 = Dot(axes=(1, 2))([yRealT, yReal])
        y2 = Dot(axes=(1, 2))([yImagT, yImag])
        y = Add()([y1, y2])

        # set diag part to zero
        yDiag = Lambda(lambda y: tf.linalg.diag_part(y))(y)
        y = Subtract()([y, tf.linalg.diag(yDiag)])

        # sum up all elements not on diagonal (penalize y1^H y2 != 0)
        y = self.w1 * tf.reduce_sum(tf.abs(y), [1, 2])

        # sum up all elements on diagonal (penalize y1^H y1 != 1)
        yDiag = self.w2 * tf.reduce_sum(tf.abs(1 - yDiag), 1)

        y = Add()([y, yDiag])

        return K.mean(y)


#****************************************#
#   orthogonalization with Gram-Schmidt  #
#****************************************#
class GramSchmidt(Layer):
    # TODO: !does not work as it should!
    """
        Defines custom layer applying the Gram-Schmidt orthogonalization.
    """

    def __init__(self, batch_size, m, n):
        super(GramSchmidt, self).__init__()
        self.batch_size = batch_size
        self.m = m
        self.n = n

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape),
                                      trainable=True)

        super(GramSchmidt, self).build(input_shape)

    def call(self, input_data):
        return gram_schmidt(input_data)


def gram_schmidt(input_data):
    yReal = Lambda(lambda y: y[:, :m])(input_data)
    yImag = Lambda(lambda y: y[:, m:])(input_data)

    yRealT = Permute((2, 1))(yReal)
    yImagT = Permute((2, 1))(yImag)

    Qreal = tf.Variable(np.zeros((batch_size, m, n)), dtype='float32')
    Qimag = tf.Variable(np.zeros((batch_size, m, n)), dtype='float32')

    for i in range(n):
        qReal = yRealT[:, i, :]
        qImag = yImagT[:, i, :]
        for j in range(i):
            y1 = Dot(axes=(1, 1))([qReal, Qreal[:, :, j]])
            y2 = Dot(axes=(1, 1))([qImag, Qimag[:, :, j]])
            rij = Add()([y1, y2])
            qReal = Subtract()([yReal[:, :, i], rij * Qreal[:, :, j]])
            qImag = Subtract()([yImag[:, :, i], rij * Qimag[:, :, j]])

        y1 = Dot(axes=(1, 1))([yRealT[:, i, :], yReal[:, :, i]])
        y2 = Dot(axes=(1, 1))([yImagT[:, i, :], yImag[:, :, i]])
        y = Add()([y1, y2])

        Qreal[:, :, i].assign(qReal / y)
        Qimag[:, :, i].assign(qImag / y)

    Q = Concatenate(axis=1)([Qreal, Qimag])

    return Q