'''
DNN playground using KERAS
test DNN model on ENST dataset
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList
from scipy.io import loadmat
from keras.models import load_model

'''
==== User input
'''
targetDrummers = ['drummer1',
                'drummer2',
                'drummer3']
parentFolder     = '../../unlabeledDrumDataset/evaluation_enst/CQT/'
saveParentFolder = '../../unlabeledDrumDataset/evaluation_enst/Activations/'
# parentFolder     = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/STFT/'
# saveParentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/Activations/'
modelpath = './models/dnn_model_1nn_50songs_bs640_CQT_smt.h5'

'''
==== File IO + testing
'''

model = load_model(modelpath)

for drummer in targetDrummers:
    #==== get STFT
    cqtpath = parentFolder + drummer + '/'
    cqtFilePathList = getFilePathList(cqtpath, 'mat')
    saveFolder = saveParentFolder + drummer + '/'


    for i in range(0, len(cqtFilePathList)):
        print 'test on file %f' % i
        filename = cqtFilePathList[i][-11:-4]
        savepath = saveFolder + filename
        tmp = loadmat(cqtFilePathList[i])
        X = np.ndarray.transpose(tmp['Xcqt'])

        '''
        ==== Testing
        '''
        X_diff = np.diff(X, axis=0)
        finalRow = np.zeros((1, np.size(X, 1)))
        X_diff = np.concatenate((X_diff, finalRow), axis=0)
        X_all = np.concatenate((X, X_diff), axis=1)

        Y = model.predict(X_all, batch_size = 32)
        all = [Y[:, 0], Y[:, 1], Y[:, 2]]
        np.save(savepath, all)


