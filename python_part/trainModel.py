'''
DNN playground using KERAS
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout

'''
==== User input
'''
targetPseudoLabels = ['200drums', 'enst', 'smt']
targetGenres = ['dance-club-play-songs',
                'hot-mainstream-rock-tracks',
                'latin-songs',
                'pop-songs',
                'r-b-hip-hop-songs']
parentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/'
savepath = './models/dnn_model.npy'

'''
==== Define DNN model
'''
model = Sequential()
model.add(Dense(1025, input_dim = 1025, init = 'normal', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, init = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, init = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(32, init = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init = 'uniform', activation = 'relu'))
model.compile(optimizer = 'adam', loss='mse', metrics = ['mae'])


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
            model.fit(X, Y, nb_epoch = 1, batch_size = 32)

'''
==== Save the trained DNN model
'''
np.save(savepath, model)