'''
DNN playground using KERAS
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

'''
==== User input
'''
targetPseudoLabels = ['200drums', 'enst', 'smt']
targetGenres = ['dance-club-play-songs',
                'hot-mainstream-rock-tracks',
                'latin-songs',
                'pop-songs',
                'r-b-hip-hop-songs']
#parentFolder = '../../unlabeledDrumDataset/activations/'
parentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/'
savepath_hh = './models/dnn_model_hh.h5'
savepath_kd = './models/dnn_model_kd.h5'
savepath_sd = './models/dnn_model_sd.h5'

'''
==== Define DNN model
'''

def createModel():
    model = Sequential()
    model.add(Dense(units = 1025, input_dim = 1025))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 512))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 512))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1))
    model.add(Activation('relu'))
    model.compile(optimizer = 'rmsprop', loss='mean_squared_error', metrics = ['mae'])
    return model

model_hh = createModel()
model_kd = createModel()
model_sd = createModel()

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

        for i in range(0, 1): #len(stftFilePathList)):
            tmp = loadmat(stftFilePathList[i])
            X = np.ndarray.transpose(tmp['X'])
            tmp = loadmat(pseudoLabelFilePathList[i])
            Y = np.ndarray.transpose(tmp['HD'])
            assert (len(X) == len(Y)), 'dimensionality mismatch between STFT and Pseudo-Labels!'

            '''
            ==== Training
            '''
            print '==== training HH ====\n'
            model_hh.fit(X, Y[:, 0], epochs = 1, batch_size = 32)
            print '==== training KD ====\n'
            model_kd.fit(X, Y[:, 1], epochs = 1, batch_size = 32)
            print '==== training SD ====\n'
            model_sd.fit(X, Y[:, 2], epochs = 1, batch_size = 32)

'''
==== Save the trained DNN model
'''
model_hh.save(savepath_hh)
model_kd.save(savepath_kd)
model_sd.save(savepath_sd)