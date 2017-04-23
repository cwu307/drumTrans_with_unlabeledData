'''
DNN playground using KERAS
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList, scaleData
from scipy.io import loadmat
from sklearn.linear_model import SGDRegressor

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
# parentFolder = '../../unlabeledDrumDataset/activations/'
parentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/'
savepath = './models/svr_model.npy'

'''
==== Define SVR model
'''
clf_hh = SGDRegressor()
clf_bd = SGDRegressor()
clf_sd = SGDRegressor()

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

        for i in range(0, 200): #len(stftFilePathList)):
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
        [y_bd_scaled, dump, dump] = scaleData(Y[:, 1])
        [y_sd_scaled, dump, dump] = scaleData(Y[:, 2])
        #y_all = np.concatenate((y_hh_scaled, y_kd_scaled, y_sd_scaled), axis=1)

        print '==== training ====\n'
        clf_hh.partial_fit(X, y_hh_scaled)
        clf_bd.partial_fit(X, y_bd_scaled)
        clf_sd.partial_fit(X, y_sd_scaled)

svrModel = [clf_hh, clf_bd, clf_sd]
'''
==== Save the trained DNN model
'''
np.save(savepath, svrModel)
