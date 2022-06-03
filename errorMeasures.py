####################################################################################################
#                                         errorMeasures.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/04/21                                                                                #
#                                                                                                  #
# Purpose: Definitions of custom error measures used to evaluate DoA estimation algorithms.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from syntheticEx import angles
from utils import permutations


#***********************************************#
#   eliminate randomness to reproduce results   #
#***********************************************#
np.random.seed(42)


#**********************#
#   simple mean rmse   #
#**********************#
def mean_naive_rmse(predDoA, trueDoA):
    """
            Calculates the mean of all the samples of a naive rmse, i.e. an rmse that
            takes the squared error by padding/truncating and then sorting the DoA.

            @param predDoA -- The estimated DoA angles in radians (same length as true DoA).
            @param trueDoA -- The ground truth DoA angles in radians.

            @returns -- The mean of the naive rmse.
    """
    num_samples = trueDoA.shape[0]

    allMSE = np.zeros(num_samples)
    for i in range(num_samples):

        # angular difference
        diff = ((np.sort(predDoA[i]) - np.sort(trueDoA[i])) + np.pi / 2) % np.pi - np.pi / 2

        # rmse
        allMSE[i] = np.mean(diff ** 2) ** (1 / 2)

    return np.mean(allMSE)


#***********************************#
#   mean minimal permutation rmse   #
#***********************************#
def mean_min_perm_rmse(predDoA, trueDoA):
    """
            Calculates the mean of all the samples of the minimal rmse of
            (all permutations of) the predicted DoA and the true DoA.

            @param predDoA -- The estimated DoA angles in radians (same length as true DoA).
            @param trueDoA -- The ground truth DoA angles in radians.

            @returns -- The mean of minimal rmse.
    """
    num_samples = trueDoA.shape[0]

    phi_min, phi_max = angles[0, 0], angles[0, -1]

    allMSE = np.zeros(num_samples)
    for i in range(num_samples):

        if trueDoA.shape[1] == 1:
            # angular difference
            diff = ((predDoA[i] - trueDoA[i]) + phi_max / 2) % (phi_max - phi_min) - phi_max / 2

            # rmse
            allMSE[i] = np.mean(diff ** 2) ** (1 / 2)

        else:
            # remove 'nan' values (use with variable number of sources)
            true = np.array(trueDoA[i][trueDoA[i] <= np.pi / 2])

            # get permutations of estimated DoA
            diffs = np.zeros(np.math.factorial(len(true)))
            for j, perm in enumerate(permutations(list(predDoA[i]))):

                # angular difference
                diff = ((perm - true) + phi_max / 2) % (phi_max - phi_min) - phi_max / 2

                # rmse
                diffs[j] = np.mean(diff ** 2) ** (1 / 2)

            # choose minimal rmse as error
            allMSE[i] = np.amin(diffs)

    return np.mean(allMSE)


#****************************************#
#   number of sources estimation error   #
#****************************************#
def num_source_err(predDoA, trueDoA):
    """
        Calculates the mean of all the samples of the difference of estimated
        number of sources and true number of sources.

        @param predDoA -- The estimated DoA angles in radians (same length as true DoA).
        @param trueDoA -- The ground truth DoA angles in radians.

        @returns -- The average difference.
    """
    num_samples = trueDoA.shape[0]

    tot = 0

    for i in range(num_samples):
        # remove 'nan' values
        true = np.array(trueDoA[i][trueDoA[i] <= np.pi / 2])
        pred = np.array(predDoA[i][predDoA[i] <= np.pi / 2])

        # count num. source diff
        tot += abs(len(true) - len(np.array(pred)))

    return tot / num_samples