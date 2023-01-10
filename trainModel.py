####################################################################################################
#                                           trainModel.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 23/03/21                                                                                #
#                                                                                                  #
# Purpose: Training of the augmentation for the MUSIC algorithm. Neural network outputs noise      #
#          subspaces directly from the measurements.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from findpeaks import findpeaks

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import Regularizer, L1, L1L2
from tensorflow.keras.utils import plot_model

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_curve, accuracy_score, classification_report
from sklearn.preprocessing import normalize, Normalizer, MinMaxScaler, StandardScaler

from scipy import linalg
from scipy import signal
from scipy.stats import laplace

from augMUSIC import augMUSIC
# from bbSynthEx import *
from beamformer import beamformer
from broadbandMUSIC import broadMUSIC
from classicMUSIC import classicMUSIC, cluster
from errorMeasures import *
from losses import *
from models import *
from cssMethod import css
from syntheticEx import *


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)
tf.random.set_seed(42)

# tf.compat.v1.enable_eager_execution()
# tf.config.run_functions_eagerly(True)


#****************************************#
#   read files (may need to alter path)  #
#****************************************#
hf = h5py.File('data/SOREQ/m18_2816_l12000_m0l_sen_rvel_12k_2ke10k.h5', 'r')

offset = 1000
dataX = np.array(hf.get('X'))[:, :, offset:snapshots + offset]
dataY = np.array(hf.get('Y'))
dataA = np.array(hf.get('A'))[:, :m]
dataE = np.array(hf.get('E'))
dataS = np.array(hf.get('S'))

# dataX, dataY = utils.shuffle(dataX, dataY)
# dataX, dataY, dataA = utils.shuffle(dataX, dataY, dataA)
dataX, dataY, dataA, dataE, dataS = utils.shuffle(dataX, dataY, dataA, dataE, dataS)

trainX_real = np.real(dataX)
trainX_imag = np.imag(dataX)

trainX = np.concatenate((trainX_real, trainX_imag), axis=1)

# trainX = dataX   # ignore imag part !!!

print("GERES TEST ERROR:", np.min(dataE), np.max(dataE), np.mean(dataE))

# # filter
# dX = []
# dY = []
# dA = []
# dE = []
# dS = []
# for i, data in enumerate(dataY):
#     if data < 180:
#         dX.append(trainX[i])
#         dY.append(dataY[i])
#         dA.append(dataA[i])
#         dE.append(dataE[i])
#         dS.append(dataS[i])
#
# trainX = np.array(dX)
# dataY = np.array(dY)
# dataA = np.array(dA)
# dataE = np.array(dE)
# dataS = np.array(dS)


#********************#
#   initialization   #
#********************#
num_samples = dataX.shape[0]
num_samples = trainX.shape[0]
n = m - dataY.shape[1]   # number of noise vectors
r = angles.shape[1]   # resolution (i.e. angle grid size)

batch_size = 16


# # remove 'nan'
# d = []
# for i in range(num_samples):
#     d.append(dataY[i, np.isfinite(dataY[i])])
#
# dataY = d

# # create batch dimension to assure same d in each batch
# trainX = trainX.reshape((num_samples // batch_size, batch_size, trainX.shape[1], trainX.shape[2]))
# dataY = dataY.reshape((num_samples // batch_size, batch_size, dataY.shape[1]))
#
# trainX, dataY = utils.shuffle(trainX, dataY)
#
# # remove batch dimension
# trainX = trainX.reshape(((num_samples // batch_size) * batch_size, trainX.shape[2], trainX.shape[3]))
# dataY = dataY.reshape(((num_samples // batch_size) * batch_size, dataY.shape[2]))


# extract features: real(Kx), imag(Kx), angle(Kx), (where Kx is covariance of X)
trainKx = np.zeros((num_samples, 3, m, m))
for i in range(num_samples):
    Kx = np.cov(dataX[i])

    trainKx[i, 0] = np.real(Kx)
    trainKx[i, 1] = np.imag(Kx)
    trainKx[i, 2] = np.angle(Kx)


#****************************#
#   build perfect spectrum   #
#****************************#
# ds = np.array([len(dataY[i][dataY[i] <= np.pi / 2]) for i in range(num_samples)])
y = ((dataY + np.pi/2) / np.pi * r).astype(int)
trainY = np.zeros((num_samples, r))
for i in tqdm(range(num_samples)):
    # d = len(y[i][y[i] <= np.pi / 2]) 
    # d = ds[i]

    # dirac impulses
    # trainY[i] = (10 ** (snr / 10)) * signal.unit_impulse(r, y[i]) + 1
    trainY[i] = signal.unit_impulse(r, y[i, :d])

    # # laplace distributions
    # for j in range(d):
    #    trainY[i] +=  100 * laplace.pdf(np.linspace(0, r, r), loc=y[i, j], scale=1)

    # # classic MUSIC spectrum
    # trainY[i] = classicMUSIC(trainX[i, :m] + 1j * trainX[i, m:], array, angles, d)[1]

# trainY = MinMaxScaler().fit_transform(trainY)


#*******************************************#
#   transform DoA angles to spectrum locs   #
#*******************************************#
# trainY = ((dataY + np.pi/2) / np.pi * r).astype(int).astype('float64')


#**************************#
#   create EVD as labels   #
#**************************#
trainEVD = np.zeros((num_samples, m, n)) + 1j * np.zeros((num_samples, m, n))
# for i in range(num_samples):
#     X = trainX[i, :m] + 1j * trainX[i, m:]
#     covariance = np.cov(X)
#     eigenvalues, eigenvectors = linalg.eig(covariance)
#
#     # the noise matrix
#     trainEVD[i] = eigenvectors[:, d:]
#     # trainEVD[i] = trainEVD[i] / np.linalg.norm(trainEVD[i])
#
# trainEVD_real = np.real(trainEVD) / np.linalg.norm(np.real(trainEVD))
# trainEVD_imag = np.imag(trainEVD) / np.linalg.norm(np.imag(trainEVD))
#
# trainEVD = np.concatenate((trainEVD_real, trainEVD_imag), axis=1)


#*************************#
#   filter measurements   #
#*************************#
# X = trainX[:, :m] + 1j * trainX[:, m:]
#
# Xft = np.fft.fft(X, n=fSamp)
#
# NUMS = 10
# trainX = np.zeros((num_samples, m, NUMS, snapshots)) + \
#          1j * np.zeros((num_samples, m, NUMS, snapshots))
# for i in tqdm(range(NUMS)):
#     res = np.zeros((num_samples, m, fSamp))+ 1j * np.zeros((num_samples, m, fSamp))
#     ind = int(np.min(fs)) + i * len(fs) // NUMS
#     res[:, :, ind:ind + len(fs) // NUMS] = Xft[:, :, ind:ind + len(fs) // NUMS]
#     ind += len(fs)
#     res[:, :, ind:ind + len(fs) // NUMS] = Xft[:, :, ind:ind + len(fs) // NUMS]
#     trainX[:, :, i, :] =  np.fft.ifft(res)[:, :, :snapshots]
#
# trainX = np.fft.ifft(Xft[:, :, 400:4400], n=fSamp)
#
# trainX_real = np.real(trainX)
# trainX_imag = np.imag(trainX)
#
# trainX = np.concatenate((trainX_real, trainX_imag), axis=1)


#****************************#
#   build steering vectors   #
#****************************#
# trainA = np.zeros((num_samples, m)) + 1j * np.zeros((num_samples, m))
# for i in range(num_samples):
#     trainA[i] = (build_steering_vect(dataA[i], dataY[i]))

# trainA_real = np.real(trainA)
# trainA_imag = np.imag(trainA)
# trainAcplx = np.concatenate((trainA_real, trainA_imag), axis=1)


#*******************#
#   normalization   #
#*******************#
dataE = dataE * np.pi / 180   # from deg to rad
dataY = dataY * np.pi / 180   # from deg to rad
dataS = dataS / np.pi * 180   # from s/deg to s/rad

# for i in range(num_samples):
#     trainX[i, :, :] = MinMaxScaler().fit_transform(trainX[i])

# for i in range(num_samples):
#     dataA[i, :, :] = MinMaxScaler().fit_transform(dataA[i])


ENTIRE = False   # set to true when testing with entire data

if ENTIRE:
    # take entire set for testing
    # testX, testY, trainDoA, testDoA, testKx = trainX, trainY, trainEVD, dataY, trainKx

    # testX, testY, testA, testE, testEVD, testDoA, testKx = trainX, trainY, dataA, dataE, trainEVD, dataY, trainKx

    testX, testY, testA, testE, testEVD, testDoA, testKx, testS = trainX, trainY, dataA, dataE, trainEVD, dataY, trainKx, dataS

else:
    # split train set for testing
    # trainX, testX, trainY, testY, trainEVD, testEVD, trainDoA, testDoA, trainKx, testKx = \
    # train_test_split(trainX, trainY, trainEVD, dataY, trainKx, test_size=0.1)

    # trainX, testX, trainY, testY, trainA, testA, trainEVD, testEVD, trainDoA, testDoA, trainKx, testKx = \
    # train_test_split(trainX, trainY, dataA, trainEVD, dataY, trainKx, test_size=0.1)

    # trainX, testX, trainY, testY, trainA, testA, trainE, testE, trainEVD, testEVD, trainDoA, testDoA, trainKx, testKx = \
    # train_test_split(trainX, trainY, dataA, dataE, trainEVD, dataY, trainKx, test_size=0.1)

    trainX, testX, trainY, testY, trainA, testA, trainE, testE, trainEVD, testEVD, trainDoA, testDoA, \
    trainKx, testKx, trainS, testS = train_test_split(trainX, trainY, dataA, dataE, trainEVD, dataY,
                                                      trainKx, dataS, test_size=0.1)

    # trainX, testX, trainDoA, testDoA = train_test_split(trainX, dataY, test_size=0.1)



if __name__ == "__main__":

    TRAIN = False   # set to true when training a model

    E2E = True   # set to true when evaluating an end2end model
    SPEC = False   # set to true when training with Cov and Spec
    CLASS = False  # set to true when training with Cov and Class
    FULL = True   # set true when outputting dmax DoA angles
    SEP = False   # set true when outputting DoA angles separated
    EST = False   # set true when training separate d estimator
    CLA = False   # set true when only training a classifier
    TWO = False   # set true when the model outputs two things
    STV = True   # set true when passing measurements and senor positions
    SLO = True   # set true when passing measurements, senor positions, and slowness

    LOSS = mape

    # # load DA-MUSIC
    # inX, outY = aug_MUSIC_soft_q_est()
    # aug = Model(inX, outY)
    # aug.load_weights("model/aug_soft_q_est.h5")
    #
    # # load classifier
    # inX, outY = separate_est_d(aug)
    # clas = Model(inX, outY)
    # clas.load_weights("model/sep_class_d2-5_l200.h5")

    # x, y = soft_q_est_d_NON_TRAIN(aug, clas)
    # model = Model(x, y)

    x, y = deep_aug_MUSIC()
    model = Model(x, y)

    if CLASS:
        LOSS = 'binary_crossentropy'
        trainX = np.swapaxes(np.swapaxes(trainKx, 1, -1), 1, 2)
        testKx = np.swapaxes(np.swapaxes(testKx, 1, -1), 1, 2)

    if TRAIN:
        if E2E: trainY, testY, LOSS = trainDoA, testDoA, perm_rmse
        if SPEC: trainX, LOSS = trainKx, 'mse'
        if EST: trainY, testY, LOSS = \
            [trainDoA, tf.math.argmax(trainDoA, axis=1) - 2],\
            [testDoA, tf.math.argmax(testDoA, axis=1) - 2], \
            [rmse_est_d_single, 'sparse_categorical_crossentropy']
        if CLA: trainY, testY, LOSS = \
            tf.math.argmax(trainDoA, axis=1) - 2, tf.math.argmax(testDoA, axis=1) - 2, \
            'sparse_categorical_crossentropy'
        elif TWO: LOSS = [rmse_est_d_single, mse_est_d_p_i]


        model.summary()
        model.compile(loss=LOSS, optimizer=Adam(lr=0.001), metrics=['accuracy'],)
        checkpoint = ModelCheckpoint(save_best_only=True,
                                     filepath='model/dam.h5',
                                     save_weights_only=True,
                                     verbose=1)
        if SPEC:
            q = 24

            trainY, testY = np.array_split(trainY, q, axis=1), np.array_split(testY, q, axis=1)
            for i in range(q): trainY[i] = StandardScaler().fit_transform(trainY[i])
            for i in range(q): testY[i] = StandardScaler().fit_transform(testY[i])

            history = model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=1,
                                validation_split=0.2, callbacks=[checkpoint], verbose=1)

            results = model.evaluate(testKx, testY, batch_size=batch_size)

        elif STV:
            if SLO:
                history = model.fit(x=(trainX, trainA, trainS), y=trainY, batch_size=batch_size, epochs=600,
                                    validation_split=0.2, callbacks=[checkpoint], verbose=1)

                results = model.evaluate((testX, testA, testS), testY, batch_size=batch_size)
            else:
                history = model.fit(x=(trainX, trainA), y=trainY, batch_size=batch_size, epochs=600,
                                    validation_split=0.2, callbacks=[checkpoint], verbose=1)

                results = model.evaluate((testX, testA), testY, batch_size=batch_size)

        else:
            history = model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=50,
                                validation_split=0.2, callbacks=[checkpoint], verbose=1)

            if CLASS: results = model.evaluate(testKx, testY, batch_size=batch_size)
            else: results = model.evaluate(testX, testY, batch_size=batch_size)

        print("TEST LOSS:", results)

    else:
        model.load_weights("model/dam.h5")
        # model.compile(loss=LOSS, optimizer=Adam(lr=0.001), metrics=['accuracy'], )
        # results = model.evaluate((testX, testA), testY, batch_size=batch_size)
        # print("TEST LOSS:", results)


    # test_legth = 10
    # testX, testDoA = testX[:test_legth, :, :], testDoA[:test_legth, :]
    # testA = testA[:test_legth, :]


    #*********************#
    #   evaluate models   #
    #*********************#
    num_samples = testX.shape[0]

    phi_min, phi_max = angles[0, 0], angles[0, -1]
    phi_span = phi_max - phi_min

    az_min, az_max = angles[0, 0], angles[0, -1]
    az_span = az_max - az_min

    el_min, el_max = angles[1, 0], angles[1, -1]
    el_span = el_max - el_min

    AugDoA = []
    AugDoAcomp = []
    AugEst = []
    ClasDoA = []
    ClasDoA2D = []
    ClasDoAcomp = []
    ClasEst = []
    BroadDoA = []
    BroadDoA2D = []
    BroadDoAcomp = []
    cssDoA = []
    cssDoAcomp = []
    BeamDoA = []
    BeamDoA2D = []
    BeamEst = []
    RanDoA = []
    ZeroDoA = []
    Geres = []
    for i in tqdm(range(num_samples)):
        d = len(testDoA[i][testDoA[i] <= np.pi / 2])

        # end2end #
        #*********#
        if E2E:
            X = np.repeat(testX[i][np.newaxis, :, :], 1, axis=0)
            A = np.repeat(testA[i][np.newaxis, :], 1, axis=0)
            if SEP: DoA = model.predict(X)
            else:
                # DoA = model.predict(X)[0]
                DoA = model.predict((X, A))[0]

        # deepMUSIC or CNN #
        #******************#
        elif SPEC or CLASS:
            X = np.repeat(testKx[i][np.newaxis, :, :], 1, axis=0)
            spectrum = np.concatenate(model.predict(X), axis=None)
            DoA, _ = signal.find_peaks(spectrum, distance=10)

            if False: # (set True for known num. sources)
                # only keep d largest peaks
                DoA = DoA[np.argsort(spectrum[DoA])[-d:]]
            else:
                # or give all peak locations in descending order
                p_i = 0.05
                DoA = DoA[spectrum[DoA] > p_i]
                DoA = DoA[np.argsort(- spectrum[DoA])]

            # transform to radians
            DoA = DoA * phi_span / r + phi_min

        # aug MUSIC # <-> ARTEFACT!
        #***********#
        else:
            X = np.repeat(testX[i][np.newaxis, :, :], 1, axis=0)
            DoA, spectrum = augMUSIC(model.predict(X), array, angles, d)

            # transform to radians
            DoA = DoA * phi_span / r + phi_min

        AugDoAcomp.append(DoA)

        # # estimate number of sources
        # inp = model.input
        # outputs = [layer.output for layer in model.layers]
        # # select output layer forming the spectrum and final layer (with DoA)
        # functor = [K.function([inp], [outputs[9]]), K.function([inp], [outputs[-1]])]
        #
        # lambdas, DoA = [f([X]) for f in functor]
        #
        # n = cluster(np.array(lambdas[0][0][0][0])).shape[0]  # estimate multiplicity of smallest eigenvalue...
        # d = array.shape[0] - n  # and get number of signal sources

        # d = np.argmax(model.predict(X)[1][0]) + 2

        if SEP: DoA = DoA[d-2][0]
        elif TWO:
            d_aug = np.argmax(model.predict(X)[1][0]) + 2
            # d_aug = np.argmax(clas.predict(X)[0]) + 2
            AugEst.append(d_aug)
            DoA = DoA[0][:d_aug]
        elif FULL: DoA = DoA[:d]

        # d = len(testDoA[i][testDoA[i] <= np.pi / 2])

        # ensure exact number of DoA are compared
        if len(DoA) < d:
            # add zero for all non-present angles
            DoA = np.append(DoA, [np.random.uniform(phi_min, phi_max)
                                  for _ in range(d - len(DoA))])
        elif len(DoA) > d:
            DoA = np.sort(DoA)
            DoA = DoA[:d]

        AugDoA.append(DoA)


        # classic MUSIC #
        #***************#
        X = testX[i, :m] + 1j * testX[i, m:]

        # filter
        FTx = np.fft.fft(X, axis=1)
        maxF = 10
        FTx = np.concatenate((FTx[:, :maxF], FTx[:, - maxF:]), axis=1)
        X = np.fft.ifft(FTx, axis=1)

        # DoAMUSIC, spectrum, d_MUSIC = classicMUSIC(X, array, angles)
        DoAMUSIC, spectrum, d_MUSIC, spectrum2D = classicMUSIC(X, testA[i], angles, testS[i], d)

        # transform to radians
        DoAMUSIC = DoAMUSIC * phi_span / r + phi_min

        ClasDoAcomp.append(DoAMUSIC)

        # ensure exact number of DoA are compared
        if len(DoAMUSIC) < d:
            # add zero for all non-present angles
            DoAMUSIC = np.append(DoAMUSIC, [np.random.uniform(phi_min, phi_max)
                                            for _ in range(d - len(DoAMUSIC))])
        elif len(DoAMUSIC) > d:
            DoAMUSIC = np.sort(DoAMUSIC)
            DoAMUSIC = DoAMUSIC[:d]

        fp = findpeaks(verbose=0, denoise=None)
        results = fp.fit(spectrum2D)

        azimuth = results['groups0'][0][0][1] * az_span / r + az_min
        elevation = results['groups0'][0][0][0] * el_span / r + el_min

        ClasDoA.append(DoAMUSIC)
        ClasDoA2D.append(azimuth)

        ClasEst.append(d_MUSIC)

        print(azimuth, testDoA[i])


        # broadband MUSIC #
        #*****************#
        X = testX[i, :m] + 1j * testX[i, m:]

        # DoAMUSIC, spectrum, _ = broadMUSIC(X, array, angles, d)
        DoAMUSIC, spectrum, _, spectrumBB2D = broadMUSIC(X, testA[i], angles, testS[i], d)

        # transform to radians
        DoAMUSIC = DoAMUSIC * phi_span / r + phi_min

        BroadDoAcomp.append(DoAMUSIC)

        # ensure exact number of DoA are compared
        if len(DoAMUSIC) < d:
            # add zero for all non-present angles
            DoAMUSIC = np.append(DoAMUSIC, [np.random.uniform(phi_min, phi_max)
                                            for _ in range(d - len(DoAMUSIC))])
        elif len(DoAMUSIC) > d:
            DoAMUSIC = np.sort(DoAMUSIC)
            DoAMUSIC = DoAMUSIC[:d]

        fp = findpeaks(limit=200, scale=True, verbose=0)
        resultsBB = fp.fit(spectrumBB2D)

        azimuthBB = resultsBB['groups0'][0][0][1] * az_span / r + az_min
        elevationBB = resultsBB['groups0'][0][0][0] * el_span / r + el_min

        print(azimuthBB, testDoA[i])

        BroadDoA.append(DoAMUSIC)
        BroadDoA2D.append(azimuthBB)


        # Beamformer #
        #************#
        X = testX[i, :m] + 1j * testX[i, m:]

        # # filter
        # FTx = np.fft.fft(X, axis=1)
        # maxF = 10
        # FTx = np.concatenate((FTx[:, :maxF], FTx[:, - maxF:]), axis=1)
        # X = np.fft.ifft(FTx, axis=1)

        # DoABF, spectrum = beamformer(X, array, angles)
        DoABF, spectrum, spectrumBF2D = beamformer(X, testA[i], angles, d)

        # transform to radians
        DoABF = DoABF * phi_span / r + phi_min

        BeamEst.append(len(DoABF))

        # ensure exact number of DoA are compared
        if len(DoABF) < d:
            # add zero for all non-present angles
            DoABF = np.append(DoABF, [np.random.uniform(phi_min, phi_max)
                                            for _ in range(d - len(DoABF))])
        elif len(DoABF) > d:
            # DoABF = np.sort(DoABF)
            DoABF = DoABF[:d]

        fp = findpeaks(verbose=0, denoise=None)
        resultsBF = fp.fit(spectrumBF2D)

        azimuthBF = resultsBF['groups0'][0][0][1] * az_span / r + az_min
        elevationBF = resultsBF['groups0'][0][0][0] * el_span / r + el_min

        BeamDoA.append(DoABF)
        BeamDoA2D.append(azimuthBF)

        print(azimuthBF, testDoA[i])


        # Random #
        #********#
        RanDoA.append(np.random.uniform(phi_min, phi_max, size=(d,)))
        ZeroDoA.append(np.zeros((d,)) + angles[0, r // 2])


        # Geres #
        #*******#
        Geres.append(testE[i])


    error = mean_min_perm_rmse
    print("MODEL TEST ERROR:", error(AugDoA, testDoA))
    print("CLASSIC MUSIC TEST ERROR:", error(ClasDoA, testDoA))
    print("CLASSIC MUSIC TEST ERROR 2D:", error(ClasDoA2D, testDoA))
    print("BROADBAND MUSIC TEST ERROR:", error(BroadDoA, testDoA))
    print("BROADBAND MUSIC TEST ERROR 2D:", error(BroadDoA2D, testDoA))
    print("BEAMFORMER TEST ERROR:", error(BeamDoA, testDoA))
    print("BEAMFORMER TEST ERROR 2D:", error(BeamDoA2D, testDoA))
    print("RANDOM TEST ERROR:", error(RanDoA, testDoA))
    print("ZERO TEST ERROR:", error(ZeroDoA, testDoA))
    print("GERES TEST ERROR:", np.mean(Geres))

    # print()
    # print("MODEL ACC")
    # print(classification_report(tf.math.argmax(testDoA, axis=1), AugEst))
    # print("CLASSIC MUSIC ACC")
    # print(classification_report(tf.math.argmax(testDoA, axis=1), ClasEst))
    # print("BEAMFORMER ACC")
    # print(classification_report(tf.math.argmax(testDoA, axis=1), BeamEst))
    # print("RANDOM ACC")
    # print(classification_report(tf.math.argmax(testDoA, axis=1), [np.random.choice([2, 3, 4, 5]) for _ in range(num_samples)]))