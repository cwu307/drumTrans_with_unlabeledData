'''
DNN playground using KERAS
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList, scaleData
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization, LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import adam, rmsprop, sgd
from keras import regularizers

'''
==== User input
'''
# targetPseudoLabels = ['200drums', 'enst', 'smt']
targetPseudoLabels = ['enst']
targetGenres = ['dance-club-play-songs',
                'hot-mainstream-rock-tracks',
                'latin-songs',
                'pop-songs',
                'r-b-hip-hop-songs']
parentFolder = '../../unlabeledDrumDataset/activations/'
# parentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/'
savepath = './models/dnn_model_1nn_50songs_bs640_ep30_drop.h5'

'''
==== Define DNN model
'''
optimizer = rmsprop(lr=0.001)

def createModel():
    model = Sequential()
    model.add(Dense(units = 1025, input_dim = 1025))
    model.add(normalization.BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(units = 512))
    model.add(Dropout(0.15))
    model.add(Activation('relu'))
    model.add(Dense(units = 32))
    model.add(Dropout(0.15))
    model.add(Activation('relu'))
    model.add(Dense(units = 3))
    model.add(Activation('sigmoid'))
    model.compile(optimizer = optimizer, loss='mse', metrics = ['mae'])
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
            ==== Training
            '''
            [y_hh_scaled, dump, dump] = scaleData(Y[:, 0])
            [y_kd_scaled, dump, dump] = scaleData(Y[:, 1])
            [y_sd_scaled, dump, dump] = scaleData(Y[:, 2])
            y_all = np.concatenate((y_hh_scaled, y_kd_scaled, y_sd_scaled), axis=1)
            print '==== training ====\n'
            model.fit(X, y_all, epochs = 30, batch_size = 640, callbacks=[tbCallBack])

'''
==== Save the trained DNN model
'''
model.save(savepath)
