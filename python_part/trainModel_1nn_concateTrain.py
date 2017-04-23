'''
DNN playground using KERAS
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList, scaleData
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization, LSTM
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import adam, rmsprop, sgd
from keras import regularizers

'''
==== User input
'''
# targetPseudoLabels = ['200drums', 'enst', 'smt']
targetPseudoLabels = ['smt', '200drums']
# targetGenres = ['dance-club-play-songs',
#                 'hot-mainstream-rock-tracks',
#                 'latin-songs',
#                 'pop-songs',
#                 'r-b-hip-hop-songs']
targetGenres = ['hot-mainstream-rock-tracks', 'pop-songs', 'r-b-hip-hop-songs', 'latin-songs']
parentFolder = '../../unlabeledDrumDataset/activations/'
# parentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/'
savepath = './models/dnn_model_1nn_stft.h5'

'''
==== Define DNN model
'''
optimizer = adam(lr=0.001)

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
    model.compile(optimizer = optimizer, loss='mse', metrics = ['mae'])
    return model

model = createModel()
tbCallBack = TensorBoard(log_dir='./graph/', histogram_freq=0, write_graph=True, write_images=True)
earlyStopCallBack = EarlyStopping(monitor='loss', min_delta=1e-6, patience=3)

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

        X = np.ndarray((0, 1025))
        Y = np.ndarray((0, 3))

        for i in range(0, 100): #len(stftFilePathList)):
            tmp = loadmat(stftFilePathList[i])
            X_song = np.ndarray.transpose(tmp['X'])
            tmp = loadmat(pseudoLabelFilePathList[i])
            Y_song = np.ndarray.transpose(tmp['HD'])
            assert (len(X) == len(Y)), 'dimensionality mismatch between STFT and Pseudo-Labels!'
            '''
            ==== Concatenating matrices
            '''
            X = np.concatenate((X, X_song), axis=0)
            Y = np.concatenate((Y, Y_song), axis=0)

        '''
        ==== Training
        '''
        [y_hh_scaled, dump, dump] = scaleData(Y[:, 0])
        [y_kd_scaled, dump, dump] = scaleData(Y[:, 1])
        [y_sd_scaled, dump, dump] = scaleData(Y[:, 2])
        y_all = np.concatenate((y_hh_scaled, y_kd_scaled, y_sd_scaled), axis=1)

        print '==== training ====\n'
        model.fit(X, y_all, epochs = 100, batch_size = 640, callbacks=[tbCallBack, earlyStopCallBack])

'''
==== Save the trained DNN model
'''
model.save(savepath)
