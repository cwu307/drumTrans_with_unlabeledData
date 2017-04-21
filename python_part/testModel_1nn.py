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
parentFolder     = '../../unlabeledDrumDataset/evaluation_enst/STFT/'
saveParentFolder = '../../unlabeledDrumDataset/evaluation_enst/Activations/'
# parentFolder     = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/STFT/'
# saveParentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/Activations/'
modelpath = './models/dnn_model_1nn_50songs_bs640_ep30_drop.h5'

'''
==== File IO + testing
'''

model = load_model(modelpath)

for drummer in targetDrummers:
    #==== get STFT
    stftpath = parentFolder + drummer + '/'
    stftFilePathList = getFilePathList(stftpath, 'mat')
    saveFolder = saveParentFolder + drummer + '/'


    for i in range(0, len(stftFilePathList)):
        print 'test on file %f' % i
        filename = stftFilePathList[i][-11:-4]
        savepath = saveFolder + filename
        tmp = loadmat(stftFilePathList[i])
        X = np.ndarray.transpose(tmp['X'])

        '''
        ==== Testing
        '''
        Y = model.predict(X, batch_size = 640)
        all = [Y[:, 0], Y[:, 1], Y[:, 2]]
        np.save(savepath, all)


