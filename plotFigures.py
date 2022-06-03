####################################################################################################
#                                          plotFigures.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/03/21                                                                                #
#                                                                                                  #
# Purpose: Plot synthetic examples to test correctness and performance of algorithms estimating    #
#          directions of arrival (DoA) of multiple signals.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import seaborn as sb

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from findpeaks import findpeaks

from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from augMUSIC import augMUSIC
from bbSynthEx import construct_broad_signal, construct_mod_ofdm_signal, construct_ofdm_signal
from beamformer import beamformer
from broadbandMUSIC import broadMUSIC
from classicMUSIC import *
from errorMeasures import mean_min_perm_rmse
from impMUSIC import impMUSIC
from models import *
from syntheticEx import *


#********************#
#   initialization   #
#********************#
np. set_printoptions(threshold=10)
np.set_printoptions(suppress=True)

doa = [-0.5, 0.5]
doa = [5]
# doa = np.pi * (np.random.rand(d) - 1/2)

# x, s = construct_signal(doa)
# x, s = construct_coherent_signal(doa)

hf = h5py.File('data/SOREQ/m18_2816_l12000_m0l_sen_rvel_12k_2ke10k.h5', 'r')


offset = 1000
dataX = np.array(hf.get('X'))[:, :, offset:snapshots + offset]
dataY = np.array(hf.get('Y'))
dataA = np.array(hf.get('A'))
dataS = np.array(hf.get('S'))

dataY = dataY * np.pi / 180
dataS = dataS / np.pi * 180

idx = 92
# idx = 111
idx = 1
x = dataX[idx, :, :]
doa = dataY[idx]
array = dataA[idx]
slow = dataS[idx]

# array = np.array([[0, i * 200/2, 0] for i in array])

# doa = [3]
# x, s = construct_signal(doa)

print(doa)

# x = MinMaxScaler().fit_transform(x)


#*************************#
#   aug MUSIC algorithm   #
#*************************#
# inX, outY = deep_aug_MUSIC_sca(10)
# model = Model(inX, outY)f
# model.load_weights("model/aug_sca_s1_d2.h5")

# model.summary()

# crate measurement for the augmentation
x_real = np.real(x)
x_imag = np.imag(x)
X = np.concatenate((x_real, x_imag), axis=0)
X = np.repeat(X[np.newaxis, :, :], 1, axis=0)

# # create cov. input for deepMUSIC
# K = np.zeros((3, m, m))
# Kx = np.cov(x)
# K[0] = np.real(Kx)
# K[1] = np.imag(Kx)
# K[2] = np.angle(Kx)


DoAOut = False   # true if the augmentation outputs DoA and not subspace
spec = False
other = False
fvdoa = False
doaverr = False
frqerr = False
evsCov = False

if DoAOut:
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    # select output layer forming the spectrum and final layer (with DoA)
    # -5 for DA-MUSIC and -9 for DA-MUSIC with iteg. d estim.
    functor = [K.function([inp], [outputs[-5]]), K.function([inp], [outputs[-1]])]

    pred_spec , DoA = [f([X]) for f in functor]
    pred_spec = pred_spec[0][0]

    # transform to spectrum indices
    DoA = ((np.array(DoA[0][0]) + np.pi / 2) / np.pi * r).astype(int) % r

elif spec:
    X = np.repeat(K[np.newaxis, :, :], 1, axis=0)
    pred_specs = model.predict(X)
    pred_spec = np.concatenate(pred_specs, axis=None)
    # DoA = []
    # for i, subspec in enumerate(pred_specs):
    #     angle, _ = signal.find_peaks(subspec[0], distance= r//36)
    #     DoA.append(angle + i * r//36)
    #
    # DoA = np.concatenate(DoA, axis=None)

    DoA, _ = signal.find_peaks(pred_spec, distance=10)

    # only keep d largest peaks
    DoA = DoA[np.argsort(pred_spec[DoA])[-d:]]

elif other:
    DoA, pred_spec = augMUSIC(model.predict(X), array, angles, d)

elif fvdoa:
    doa = [[1]]

    # build frequency range for DoA
    frqs = range(0, 1000, 50)
    res = []
    resFil = []
    resBB = []
    resBBFil = []
    resDAM = []
    resDAMFil = []
    for f in tqdm(frqs):
        inc, _ = construct_broad_signal(np.array(doa), np.array([f + 1]))

        DoA, _, _ = classicMUSIC(inc, array, angles, len(doa))
        DoA = DoA * np.pi / r - np.pi / 2
        res.append(DoA)

        DoA, _, s = broadMUSIC(inc, array, angles, len(doa))
        DoA = DoA * np.pi / r - np.pi / 2
        resBB.append(DoA)

        x_real = np.real(inc)
        x_imag = np.imag(inc)
        X = np.concatenate((x_real, x_imag), axis=0)
        X = np.repeat(X[np.newaxis, :, :], 1, axis=0)

        DoA = model.predict(X)[0]
        resDAM.append(DoA[1])

        # demodulation
        incFT = np.fft.fft(inc, n=snapshots, axis=1)

        fmax = snapshots // 2
        incFT  = np.concatenate((incFT[:, f:fmax], np.zeros((m, fmax - (fmax - f))),
                                np.fliplr(incFT[:, :f]), incFT[:, fmax:]), axis=-1)

        inc = np.fft.ifft(incFT[:, :snapshots], axis=1)
        # inc = inc * np.exp(1j * 2 * np.pi * (f + 1) * np.array(range(inc.shape[1])))

        DoA, _, _ = classicMUSIC(inc, array, angles, len(doa))
        DoA = DoA * np.pi / r - np.pi / 2
        resFil.append(DoA)

        fs = np.linspace(0, 100, 100, endpoint=False)
        DoA, _, s = broadMUSIC(inc, array, angles, len(doa))
        DoA = DoA * np.pi / r - np.pi / 2
        resBBFil.append(DoA)

        x_real = np.real(inc)
        x_imag = np.imag(inc)
        X = np.concatenate((x_real, x_imag), axis=0)
        X = np.repeat(X[np.newaxis, :, :], 1, axis=0)

        DoA = model.predict(X)[0]
        resDAMFil.append(DoA[1])

elif doaverr:
    doas = np.linspace(- np.pi / 2, np.pi / 2, 6)
    # doas = np.linspace(- 1, 1, 31)

    resF1 = []
    resF2 = []
    resF3 = []
    for doa in tqdm(doas):
        f = 250
        doa = [doa]

        inc, _ = construct_broad_signal(np.array(doa), np.array([f + 1]))

        # # filter
        # incFT = np.fft.fft(inc, n=snapshots, axis=1)
        # incFFT = np.zeros(incFT.shape)
        # incFFT[:, f] = incFT[:, f]
        # inc = np.fft.ifft(incFT, n=snapshots, axis=1)

        # DoA, _ = impMUSIC(inc, array, angles, len(doa), f=f+1)
        # DoA = DoA * np.pi / r - np.pi / 2
        # resF1.append(abs((DoA - doa)))

        x_real = np.real(inc)
        x_imag = np.imag(inc)
        X = np.concatenate((x_real, x_imag), axis=0)
        X = np.repeat(X[np.newaxis, :, :], 1, axis=0)

        inX, outY = deep_aug_MUSIC_sca(10, f)
        model = Model(inX, outY)
        model.load_weights("model/aug_sca_s1_d2.h5")

        DoA = model.predict(X)[0]
        resF1.append(DoA[1])

        f = 500
        inc, _ = construct_broad_signal(np.array(doa), np.array([f + 1]))

        # DoA, _ = impMUSIC(inc, array, angles, len(doa), f=f+1)
        # DoA = DoA * np.pi / r - np.pi / 2
        # resF2.append(abs((DoA - doa)))

        inX, outY = deep_aug_MUSIC_sca(10, f)
        model = Model(inX, outY)
        model.load_weights("model/aug_sca_s1_d2.h5")

        x_real = np.real(inc)
        x_imag = np.imag(inc)
        X = np.concatenate((x_real, x_imag), axis=0)
        X = np.repeat(X[np.newaxis, :, :], 1, axis=0)

        DoA = model.predict(X)[0]
        resF2.append(DoA[1])

        f = 750
        inc, _ = construct_broad_signal(np.array(doa), np.array([f + 1]))

        # DoA, _ = impMUSIC(inc, array, angles, len(doa), f=f+1)
        # DoA = DoA * np.pi / r - np.pi / 2
        # resF3.append(abs((DoA - doa)))

        x_real = np.real(inc)
        x_imag = np.imag(inc)
        X = np.concatenate((x_real, x_imag), axis=0)
        X = np.repeat(X[np.newaxis, :, :], 1, axis=0)

        inX, outY = deep_aug_MUSIC_sca(10, f)
        model = Model(inX, outY)
        model.load_weights("model/aug_sca_s1_d2.h5")

        DoA = model.predict(X)[0]
        resF3.append(DoA[1])

elif frqerr:
    doa = [- np.pi / 4, np.pi / 4]

    # build frequency range for DoA
    frqs = range(0, 1000, 50)
    frq1 = 500
    res = []
    for f in tqdm(frqs):
        inc = construct_broad_signal(np.array(doa), np.array([frq1, f]))
        incFT = np.fft.fft(inc, n=snapshots, axis=1)

        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(incFT[0]))

        if f == frq1:
            DoA, _ = impMUSIC(inc, array, angles, len(doa), f=f)
            DoA = DoA * np.pi / r - np.pi / 2
            res.append(DoA)

        else:
            # filter
            incFFT = np.zeros(incFT.shape)
            incFFT[:, frq1-3:frq1+3] = incFT[:, frq1-3:frq1+3]
            incF = np.fft.ifft(incFFT, n=snapshots, axis=1)

            plt.figure(figsize=(8, 4))
            plt.plot(np.abs(incFFT[0]))

            DoA1, _ = impMUSIC(incF, array, angles, 1, f=frq1)
            DoA1 = DoA1 * np.pi / r - np.pi / 2

            # filter
            incFFT = np.zeros(incFT.shape)
            incFFT[:, f-3:f+3] = incFT[:, f-3:f+3

                                 ]
            incF = np.fft.ifft(incFFT, n=snapshots, axis=1)

            plt.figure(figsize=(8, 4))
            plt.plot(np.abs(incFFT[0]))

            DoA2, _ = impMUSIC(incF, array, angles, 1, f=f)
            DoA2 = DoA2 * np.pi / r - np.pi / 2
            res.append([DoA1, DoA2])

elif evsCov:
    # estimate number of sources
    inp = model.input
    outputs = [layer.output for layer in model.layers]

    # select output layer forming the spectrum and final layer (with DoA)
    functor = [K.function([inp], [outputs[9]]), K.function([inp], [outputs[8]]), K.function([inp], [outputs[-1]])]

    lambdas, covariance, DoA = [f([X]) for f in functor]
    lambdas = lambdas[0][0][0][0]
    covariance = covariance[0][0]

    # model-based
    covariance = np.cov(x)
    lambdas, eigenvectors = linalg.eig(covariance)


# filter
FTx = np.fft.fft(x, axis=1)
maxF = 10
FTx = np.concatenate((FTx[:, :maxF], FTx[:, - maxF:]), axis=1)
xFil = np.fft.ifft(FTx, axis=1)



#*****************************#
#   classic MUSIC algorithm   #
#*****************************#
DoAMUSIC, spectrum, _, spectrum2D = classicMUSIC(xFil, array, angles, slow, d)
# DoA, pred_spec = testMUSIC(x, array, angles, d)


#****************#
#   beamformer   #
#****************#
DoABF, spectrumBF, spectrumBF2D = beamformer(x, array, angles, slow, d)



#********************#
#   broadband MUSIC  #
#********************#
DoABBMUSIC, spectrumBB, _, spectrumBB2D = broadMUSIC(x, array, angles, slow, d)



# fp = findpeaks(limit=200, scale=True)
fp = findpeaks(lookahead=20)

resultsBF = fp.fit(spectrumBF2D)
results = fp.fit(spectrum2D)
resultsBB = fp.fit(spectrumBB2D)

az_min, az_max = angles[0, 0], angles[0, -1]
az_span = az_max - az_min

el_min, el_max = angles[1, 0], angles[1, -1]
el_span = el_max - el_min

azimuthBF = resultsBF['groups0'][0][0][1] * az_span / r + az_min
elevationBF = resultsBF['groups0'][0][0][0] * el_span / r + el_min

print(DoABF[0] * np.pi / 180, azimuthBF, elevationBF)


azimuth = results['groups0'][0][0][1] * az_span / r + az_min
elevation = results['groups0'][0][0][0] * el_span / r + el_min

print(DoAMUSIC[0] * np.pi / 180, azimuth, elevation)


azimuthBB = resultsBB['groups0'][0][0][1] * az_span / r + az_min
elevationBB = resultsBB['groups0'][0][0][0] * el_span / r + el_min

print(DoABBMUSIC[0] * np.pi / 180, azimuthBB, elevationBB)


#********************#
#   visualize data   #
#********************#
def visData():

    plt.figure(figsize=(5, 4))
    plt.tight_layout()
    plt.plot(x[12])

    plt.figure(figsize=(5, 4))
    plt.tight_layout()
    plt.plot(np.fft.fft(x[12])[:len(x[12]) // 2])
    plt.xlabel('Frequency [Hz]')


#***********************#
#   plot BF vs. MUSIC   #
#***********************#
def plotBFvMUSIC():
    plt.figure(figsize=(5, 4))
    plt.tight_layout()
    # plt.plot(angles[0], spectrumBF, 'k--')
    plt.plot(angles[0], spectrumBF)
    plt.plot(angles[0, DoABF], spectrumBF[DoABF], 'x')
    plt.plot(doa, 0,'*')
    # plt.plot(angles[0, DoABF], spectrumBF[DoABF], color='grey', linestyle='', marker='x')
    # plt.plot(angles[0, DoAMUSIC], spectrum[DoAMUSIC], 'x')
    plt.xlabel('Azimuth angle (rad)')
    plt.ylabel('Spatial spectrum')
    plt.legend(['Beamformer', 'Estimated DoA', 'Actual DoA'])


#************************#
#   plot classic MUSIC   #
#************************#
def plotMUSIC():
    plt.figure(figsize=(8, 4))
    plt.tight_layout()
    plt.plot(angles[0], spectrum)
    # plt.plot(angles[0, DoAMUSIC], spectrum[DoAMUSIC], 'x', color='tab:orange', markersize=10.0, mew=3.0)
    plt.plot(angles[0, DoAMUSIC], spectrum[DoAMUSIC], 'x')
    plt.plot(doa, 0, '*')
    plt.xlabel('Azimuth angle [rad]')
    plt.ylabel('Spatial spectrum')
    # plt.legend(['MUSIC spectrum', 'Estimated DoA'])
    plt.legend(['Classic MUSIC', 'Estimated DoA', 'Actual DoA'])


#************************#
#   plot classic MUSIC   #
#************************#
def plotBBMUSIC():
    plt.figure(figsize=(8, 4))
    plt.tight_layout()
    plt.plot(angles[0], spectrumBB)
    # plt.plot(angles[0, DoAMUSIC], spectrum[DoAMUSIC], 'x', color='tab:orange', markersize=10.0, mew=3.0)
    plt.plot(angles[0, DoABBMUSIC], spectrumBB[DoABBMUSIC], 'x')
    plt.plot(doa, 0, '*')
    plt.xlabel('Azimuth angle [rad]')
    plt.ylabel('Spatial spectrum')
    # plt.legend(['MUSIC spectrum', 'Estimated DoA'])
    plt.legend(['Broadband MUSIC', 'Estimated DoA', 'Actual DoA'])


#************************#
#   plot est. spectrum   #
#************************#
def plotAugMUSIC():
    plt.figure(figsize=(8, 4))
    plt.plot(angles[0], pred_spec)
    # plt.plot(range(16), pred_spec)
    plt.plot(angles[0, DoA], pred_spec[DoA],'x', color='tab:orange', markersize=10.0, mew=3.0)
    # plt.plot(doa, [0 for i in range(d)],'*')
    plt.xlabel('Azimuth angle (rad)')
    plt.ylabel('Spatial spectrum')
    plt.legend(['DA-MUSIC spectrum', 'Estimated DoA'])
    # plt.legend(['Aug MUSIC', 'Estimated DoA', 'Actual DoA'])


#****************************#
#   plot frq vs single DoA   #
#****************************#
def plotFreqvDoA():
    plt.figure(figsize=(8, 4))
    plt.plot(frqs, [doa[0] for _ in range(len(frqs))], 'k-x')
    # plt.plot(frqs, res, 'g--^', zorder=5)
    # plt.plot(frqs, resFil, '--o', color='springgreen')
    plt.plot(frqs, resBB, '--', color='tab:purple')
    plt.plot(frqs, resBBFil, '--o', color='mediumorchid')
    plt.plot(frqs, resDAM, 'b--*')
    plt.plot(frqs, resDAMFil, '--v', color='dodgerblue')
    plt.xlabel('frequencies [Hz]')
    plt.ylabel('estimated DoA')

    # plt.legend(['True DoA', 'MUSIC', 'MUSIC (Demodulated)', 'BB MUSIC', 'BB MUSIC (Demodulated)', 'DA-MUSIC', 'DA-MUSIC (Demodulated)'])
    # plt.legend(['True DoA', 'MUSIC', 'MUSIC (Demodulated)','DA-MUSIC', 'DA-MUSIC (Demodulated)'])
    plt.legend(['True DoA', 'BB MUSIC', 'BB MUSIC (Demodulated)', 'DA-MUSIC', 'DA-MUSIC (Demodulated)'])
    # plt.legend(['True DoA', 'Classic MUSIC', 'BB MUSIC'])


#************************#
#   plot DoAs vs error   #
#************************#
def plotDoAvError():
    plt.figure(figsize=(8, 4))
    plt.plot(doas, resF1, 'b-+')
    plt.plot(doas, resF2, 'g-x')
    plt.plot(doas, resF3, 'r-p')
    # plt.plot(doas, resDAMFil, '-+', color='tab:purple')
    plt.xlabel('DoA [Rad]')
    plt.ylabel('RMSE')
    # plt.title('Error localizing 1 source.')

    plt.legend(['DA-MUSIC (f=250Hz)', 'DA-MUSIC (f=500Hz)', 'DA-MUSIC (f=750Hz)'])


#***********************#
#   plot frq vs error   #
#***********************#
def plotFreqvError():
    plt.figure(figsize=(8, 4))
    plt.plot(frq1, doa[0], 'r-*')
    plt.plot(frqs, [doa[1] for _ in range(len(frqs))], 'k-*')
    plt.plot(frqs, [elem[0] for elem in res], 'b-+')

    fs = []
    rs = []
    for i, elem in enumerate(res):
        if elem[1]:
            fs.append(frqs[i])
            rs.append(elem[1])

    plt.plot(fs, rs, 'g-x')
    plt.xlabel('Frequencies [Hz]')
    plt.ylabel('Error')

    plt.legend(['True DoA 1', 'True DoA 2', 'MUSIC (DoA 1)', 'MUSIC (DoA 2)'])
    # plt.legend(['True DoA', 'Classic MUSIC', 'BB MUSIC'])


#**********************#
#   plot eigenvalues   #
#**********************#
def plotEVs():
    plt.figure(figsize=(4, 4))
    plt.tight_layout()
    plt.scatter(np.real(lambdas), np.imag(lambdas), alpha=0.7)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
    plt.xlabel('real part')
    plt.ylabel('imag part')


#**********************#
#   plot eigenvalues   #
#**********************#
def plotCovs():
    # plt.figure(figsize=(11, 9))
    sb.heatmap(np.abs(covariance), cmap="Blues")


def plotBF2D():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.set_xlabel('Azimuth [rad]')
    ax.set_xlabel('Wavelength [m]')
    ax.set_ylabel('Elevation [rad]')
    X, Y = np.meshgrid(angles[0], angles[1])
    surf = ax.plot_surface(X, Y, spectrumBF2D, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    doaLine = np.empty((angles.shape[1], angles.shape[1]))
    doaLine[:, int(doa[0] / np.pi * 180)] = np.max(spectrumBF2D)
    doaLine = np.empty(angles.shape[1])
    doaLine[:] = np.mean(spectrumBF2D)

    doaX = np.empty(angles.shape[1])
    doaX[:] = doa[0]

    # ax.plot(doaX, angles[1], doaLine, 'ro', label='True DoA', zorder=10, markersize=2)


def plotMUSIC2D():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.set_xlabel('Azimuth [rad]')
    ax.set_xlabel('Wavelength [m]')
    ax.set_ylabel('Elevation [rad]')

    X, Y = np.meshgrid(angles[0], angles[1])
    surf = ax.plot_surface(X, Y, spectrum2D, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    doaLine = np.empty((angles.shape[1], angles.shape[1]))
    doaLine[:, int(doa[0] / np.pi * 180)] = np.max(spectrum2D)
    doaLine = np.empty(angles.shape[1])
    doaLine[:] = np.mean(spectrum2D)

    doaX = np.empty(angles.shape[1])
    doaX[:] = doa[0]

    # ax.plot(doaX, angles[1], doaLine, 'ro', label='True DoA', zorder=10, markersize=2)



def plotBBMUSIC2D():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # ax.set_xlabel('Azimuth [rad]')
    ax.set_xlabel('Wavelength [m]')
    ax.set_ylabel('Elevation [rad]')

    X, Y = np.meshgrid(angles[0], angles[1])
    surf = ax.plot_surface(X, Y, spectrumBB2D, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    doaLine = np.empty((angles.shape[1], angles.shape[1]))
    doaLine[:, int(doa[0] / np.pi * 180)] = np.max(spectrumBB2D)
    doaLine = np.empty(angles.shape[1])
    doaLine[:] = np.mean(spectrumBB2D)

    doaX = np.empty(angles.shape[1])
    doaX[:] = doa[0]

    # ax.plot(doaX, angles[1], doaLine, 'ro', label='True DoA', zorder=10, markersize=2)


#***************************#
#   plot brodband signals   #
#***************************#
def plotBBsignal():
    fcs = np.random.choice(fs, d)
    fcs = [200, 600]
    x, s = construct_broad_signal(doa, fcs)
    # x = construct_ofdm_signal(doa, fcs)
    # x = construct_mod_ofdm_signal(doa, fcs)
    xFT = np.fft.fft(x)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1000), np.abs(xFT[0, :1000]))
    plt.plot(fcs[0], np.abs(s[0, fcs[0]]))
    plt.plot(fcs[1], np.abs(s[0, fcs[1]]))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signal amplitude')




# plt.style.use(['grid', 'science', 'no-latex'])
#
#
# plt.style.use(['science', 'no-latex'])
# plt.rcParams['font.size'] = 14
# plt.rcParams['legend.fontsize'] = 11
#
# plt.rcParams['lines.linewidth'] = 2
# # plt.rcParams['lines.linestyle'] = '--'
#
# plt.rcParams['xtick.major.size'] = 4
# plt.rcParams['xtick.major.width'] = 1
# plt.rcParams['xtick.minor.size'] = 2
# plt.rcParams['xtick.minor.width'] = 1
# plt.rcParams['ytick.major.size'] = 4
# plt.rcParams['ytick.major.width'] = 1
# plt.rcParams['ytick.minor.size'] = 2
# plt.rcParams['ytick.minor.width'] = 1
# plt.rcParams['axes.linewidth'] = 1
#
# plt.rcParams['axes.grid.which'] = 'major'
#
# plt.rcParams['legend.frameon'] = True
# plt.rcParams["legend.framealpha"] = 0.5
# plt.rcParams['legend.labelspacing'] = 0.5
# plt.rcParams["legend.edgecolor"] = '1.0'
# # plt.rcParams['legend.borderaxespad'] = 1.0
#
# # plt.rcParams["legend.loc"] = 'upper right'
# plt.rcParams["legend.loc"] = 'upper center'

visData()
plotBFvMUSIC()
plotMUSIC()
plotBBMUSIC()
# plotBBMUSIC()
# plotAugMUSIC()
# plotFreqvDoA()
# plotDoAvError()
# plotFreqvError()
# plotEVs()
# plotCovs()
# plotBBsignal()
plotBF2D()
plotMUSIC2D()
plotBBMUSIC2D()

plt.show()