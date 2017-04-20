'''
DNN playground using KERAS
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList, scaleData
from transcriptUtil import medianThreshold, thresNvt, findPeaks, onset2BinaryVector
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization, LSTM
from keras.callbacks import TensorBoard

'''
==== User input
'''
targetPseudoLabels = ['200drums', 'enst', 'smt']
targetGenres = ['dance-club-play-songs',
                'hot-mainstream-rock-tracks',
                'latin-songs',
                'pop-songs',
                'r-b-hip-hop-songs']
parentFolder = '../../unlabeledDrumDataset/activations/'
# parentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/'
savepath = './models/dnn_model_1nn_50songs_bs64_thres.h5'

'''
==== Define DNN model
'''

def createModel():
    model = Sequential()
    model.add(Dense(units = 1025, input_dim = 1025))
    model.add(normalization.BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(units = 512))
    model.add(Activation('relu'))
    model.add(Dense(units = 32))
    model.add(Activation('relu'))
    model.add(Dense(units = 3))
    model.add(Activation('sigmoid'))
    model.compile(optimizer = 'rmsprop', loss='binary_crossentropy', metrics = ['mae', 'accuracy'])
    return model

model = createModel()
tbCallBack = TensorBoard(log_dir='./graph/', histogram_freq=0, write_graph=True, write_images=True)

'''
==== File IO + training
'''
for method in targetPseudoLabels:
    for genre in targetGenres:
        #==== get STFT & pseudo label
        stftpath = parentFolder + 'STFT/' + genre + '/'
        stftFilePathList = getFilePathList(stftpath, 'mat')
        pseudoLabelPath = parentFolder + method + '/' + genre + '/'
        pseudoLabelFilePathList = getFilePathList(pseudoLabelPath, 'mat')

        for i in range(0, 10): #len(stftFilePathList)):
            tmp = loadmat(stftFilePathList[i])
            X = np.ndarray.transpose(tmp['X'])
            tmp = loadmat(pseudoLabelFilePathList[i])
            Y = np.ndarray.transpose(tmp['HD'])
            assert (len(X) == len(Y)), 'dimensionality mismatch between STFT and Pseudo-Labels!'

            '''
            ==== thresholding parameters
            '''
            hopSize = 512.00
            fs = 44100.00
            order = round(0.1 / (hopSize / fs))
            offsets = [0.06, 0.12, 0.24]
            for j in range(0, len(offsets)):

                offset = offsets[j]
                '''
                ==== thresholding
                '''
                thresCurve_hh = medianThreshold(Y[:, 0], order, offset)
                thresCurve_bd = medianThreshold(Y[:, 1], order, offset)
                thresCurve_sd = medianThreshold(Y[:, 2], order, offset)

                # thresNvt_hh = thresNvt(Y[:, 0], thresCurve_hh)
                # thresNvt_bd = thresNvt(Y[:, 1], thresCurve_bd)
                # thresNvt_sd = thresNvt(Y[:, 2], thresCurve_sd)

                dump, onsetInSec_hh = findPeaks(Y[:, 0], thresCurve_hh, fs, hopSize)
                dump, onsetInSec_bd = findPeaks(Y[:, 1], thresCurve_bd, fs, hopSize)
                dump, onsetInSec_sd = findPeaks(Y[:, 2], thresCurve_sd, fs, hopSize)

                onsetInBinary_hh = onset2BinaryVector(onsetInSec_hh, len(Y[:, 0]), hopSize, fs)
                onsetInBinary_bd = onset2BinaryVector(onsetInSec_bd, len(Y[:, 1]), hopSize, fs)
                onsetInBinary_sd = onset2BinaryVector(onsetInSec_sd, len(Y[:, 2]), hopSize, fs)

                '''
                ==== Training
                '''
                # [y_hh_scaled, dump, dump] = scaleData(thresNvt_hh)
                # [y_kd_scaled, dump, dump] = scaleData(thresNvt_bd)
                # [y_sd_scaled, dump, dump] = scaleData(thresNvt_sd)

                # y_all = np.concatenate((y_hh_scaled, y_kd_scaled, y_sd_scaled), axis=1)

                y_all = np.concatenate((onsetInBinary_hh, onsetInBinary_bd, onsetInBinary_sd), axis=1)
                print '==== training ====\n'
                model.fit(X, y_all, epochs = 30, batch_size = 64, callbacks=[tbCallBack])

'''
==== Save the trained DNN model
'''
model.save(savepath)
