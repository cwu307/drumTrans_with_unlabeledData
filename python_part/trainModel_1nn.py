'''
DNN playground using KERAS
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList, scaleData
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization

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
savepath = './models/dnn_model_1nn.h5'

'''
==== Define DNN model
'''

def createModel():
    model = Sequential()
    model.add(Dense(units = 1025, input_dim = 1025))
    model.add(normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units = 512))
    model.add(Activation('relu'))
    model.add(Dense(units = 32))
    model.add(Activation('relu'))
    model.add(Dense(units = 3))
    model.add(Activation('sigmoid'))
    model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics = ['mae'])
    return model

model = createModel()

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

        for i in range(0, len(stftFilePathList)):
            tmp = loadmat(stftFilePathList[i])
            X = np.ndarray.transpose(tmp['X'])
            tmp = loadmat(pseudoLabelFilePathList[i])
            Y = np.ndarray.transpose(tmp['HD'])
            assert (len(X) == len(Y)), 'dimensionality mismatch between STFT and Pseudo-Labels!'

            '''
            ==== Training
            '''
            [y_hh_scaled, dump, dump] = scaleData(Y[:, 0])
            [y_kd_scaled, dump, dump] = scaleData(Y[:, 1])
            [y_sd_scaled, dump, dump] = scaleData(Y[:, 2])
            y_all = np.concatenate((y_hh_scaled, y_kd_scaled, y_sd_scaled), axis=1)
            print '==== training ====\n'
            model.fit(X, y_all, epochs = 30, batch_size = 32)

'''
==== Save the trained DNN model
'''
model.save(savepath)
