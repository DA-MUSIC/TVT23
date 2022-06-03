####################################################################################################
#                                           soreqData.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/09/21                                                                                #
#                                                                                                  #
# Purpose: Create a dataset from the SOREQ data recordings.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from scipy.io import wavfile, loadmat


#*********************#
#   initializations   #
#*********************#
path2data = '../SOREQ/matFiles/'

data_length = 12000
snapshots = 12000

split = data_length // snapshots

doa = []
err = []
recording = []
slow = []
steering = []


#****************************************#
#   load recordings and create dataset   #
#****************************************#
fnames = os.listdir(path2data)

for name in tqdm(fnames):

    data = loadmat(path2data + name[:-4] + '.mat')

    events = data['doa'].reshape(-1, 1)
    errors = data['err'].reshape(-1, 1)

    for i in range(len(events)):
        for j in range(split):
            if errors[i] < 5:
                try:
                    slow.append(float(data['slow'][i]))
                    recording.append(data['data'][i, :18, :data_length][:, j * snapshots:(j + 1) * snapshots])
                    doa.append(events[i])
                    err.append(errors[i])
                    steering.append(data['distances'][i, :18])
                except: print('ohhh')


doa = np.array(doa)
err = np.array(err)
recording = np.array(recording)
slow = np.array(slow)
steering = np.array(steering)

print(doa.shape)
print(err.shape)
print(recording.shape)
print(slow.shape)
print(steering.shape)


#******************#
#   save dataset   #
#******************#
add = 'm18_elt5_'
method = 'm0l_sen_rvel'
encoding = '12k_2ke10k'
name = 'SOREQ/' + add + str(doa.shape[0]) + '_l' + str(snapshots) + '_' + \
       method + '_' + encoding

hf = h5py.File('data/' + name + '.h5', 'w')
hf.create_dataset('A', data=steering)
hf.create_dataset('E', data=err)
hf.create_dataset('S', data=slow)
hf.create_dataset('X', data=recording)
hf.create_dataset('Y', data=doa)
hf.close()