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
from keras.optimizers import adam, rmsprop

'''
==== User input
'''
# targetPseudoLabels = ['200drums', 'enst', 'smt']
targetPseudoLabels = ['200drums', 'smt']
targetGenres = ['dance-club-play-songs',
                'hot-mainstream-rock-tracks',
                'latin-songs',
                'pop-songs',
                'r-b-hip-hop-songs']
parentFolder = '../../unlabeledDrumDataset/activations/'
# parentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/'
savepath = './models/dnn_model_1nn_50songs_bs640_CQT_enst_2teachers.h5'

'''
==== Define DNN model
'''
rmspropOpt = rmsprop(lr=0.001)

def createModel():
    model = Sequential()
    model.add(Dense(units = 242, input_dim = 242))
    model.add(normalization.BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dense(units = 128))
    model.add(normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units = 32))
    #model.add(normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units = 3))
    #model.add(normalization.BatchNormalization())
    model.add(Activation('sigmoid'))
    model.compile(optimizer = rmspropOpt, loss='mse', metrics = ['mae'])
    return model

model = createModel()
tbCallBack = TensorBoard(log_dir='./graph/', histogram_freq=0, write_graph=True, write_images=True)


'''
==== File IO + file concatenation
'''

X = np.ndarray((0, 121))
Y = np.ndarray((0, 3))

for method in targetPseudoLabels:
    for genre in targetGenres:
        #==== get STFT & pseudo label
        cqtpath = parentFolder + 'CQT/' + genre + '/'
        cqtFilePathList = getFilePathList(cqtpath, 'mat')
        pseudoLabelPath = parentFolder + method + '/' + genre + '/'
        pseudoLabelFilePathList = getFilePathList(pseudoLabelPath, 'mat')


        for i in range(0, 10): #len(stftFilePathList)):
            tmp = loadmat(cqtFilePathList[i])
            X_song = np.ndarray.transpose(tmp['Xcqt'])
            tmp = loadmat(pseudoLabelFilePathList[i])
            Y_song = np.ndarray.transpose(tmp['HD'])
            assert (len(X_song) == len(Y_song)), 'dimensionality mismatch between CQT and Pseudo-Labels!'

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

#==== feeding CQT and delta CQT
X_diff = np.diff(X, axis=0)
finalRow = np.zeros((1, np.size(X, 1)))
X_diff = np.concatenate((X_diff, finalRow), axis=0)
X_all = np.concatenate((X, X_diff), axis=1)

model.fit(X_all, y_all, epochs = 50, batch_size = 640, callbacks=[tbCallBack], shuffle=False)

'''
==== Save the trained DNN model
'''
model.save(savepath)
