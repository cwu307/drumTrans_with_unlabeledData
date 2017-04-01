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
parentFolder = '../../unlabeledDrumDataset/evaluation_enst/STFT/'
saveParentFolder = '../../unlabeledDrumDataset/evaluation_enst/Activations/'
modelpath_hh = './models/dnn_model_hh.h5'
modelpath_kd = './models/dnn_model_kd.h5'
modelpath_sd = './models/dnn_model_sd.h5'

'''
==== File IO + testing
'''

model_hh = load_model(modelpath_hh)
model_kd = load_model(modelpath_kd)
model_sd = load_model(modelpath_sd)

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
        Y_hh = model_hh.predict(X, batch_size = 32)
        Y_kd = model_kd.predict(X, batch_size=32)
        Y_sd = model_sd.predict(X, batch_size=32)
        all = [Y_hh, Y_kd, Y_sd]
        np.save(savepath, all)


