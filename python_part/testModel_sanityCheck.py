'''
DNN playground using KERAS
test DNN model on ENST dataset
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList, scaleData
from scipy.io import loadmat
from keras.models import load_model


'''
==== User input
'''
targetDrummers = ['drummer1',
                'drummer2',
                'drummer3']
parentFolder     = '../../unlabeledDrumDataset/evaluation_enst/200drums/'
parentFolder2     = '../../unlabeledDrumDataset/evaluation_enst/enst/'
parentFolder3     = '../../unlabeledDrumDataset/evaluation_enst/smt/'
saveParentFolder = '../../unlabeledDrumDataset/evaluation_enst/Activations/'
# parentFolder     = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/STFT/'
# saveParentFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/Activations/'

'''
==== File IO + testing
'''
for drummer in targetDrummers:
    #==== get STFT
    stftpath = parentFolder + drummer + '/'
    stftFilePathList = getFilePathList(stftpath, 'mat')
    stftpath2 = parentFolder2 + drummer + '/'
    stftFilePathList2 = getFilePathList(stftpath2, 'mat')
    stftpath3 = parentFolder3 + drummer + '/'
    stftFilePathList3 = getFilePathList(stftpath3, 'mat')
    saveFolder = saveParentFolder + drummer + '/'


    for i in range(0, len(stftFilePathList)):
        print 'test on file %f' % i
        filename = stftFilePathList[i][-11:-4]
        savepath = saveFolder + filename
        tmp = loadmat(stftFilePathList[i])
        Y1 = np.ndarray.transpose(tmp['HD'])

        filename2 = stftFilePathList2[i][-11:-4]
        tmp = loadmat(stftFilePathList2[i])
        Y2 = np.ndarray.transpose(tmp['HD'])

        filename3 = stftFilePathList3[i][-11:-4]
        tmp = loadmat(stftFilePathList3[i])
        Y3 = np.ndarray.transpose(tmp['HD'])

        '''
        ==== scale the data
        '''
        [y1_0, dump, dump] = scaleData(Y1[:, 0])
        [y1_1, dump, dump] = scaleData(Y1[:, 1])
        [y1_2, dump, dump] = scaleData(Y1[:, 2])

        [y2_0, dump, dump] = scaleData(Y2[:, 0])
        [y2_1, dump, dump] = scaleData(Y2[:, 1])
        [y2_2, dump, dump] = scaleData(Y2[:, 2])

        [y3_0, dump, dump] = scaleData(Y3[:, 0])
        [y3_1, dump, dump] = scaleData(Y3[:, 1])
        [y3_2, dump, dump] = scaleData(Y3[:, 2])

        hh = np.add(np.add(y1_0, y2_0), y3_0)
        bd = np.add(np.add(y1_1, y2_1), y3_1)
        sd = np.add(np.add(y1_2, y2_2), y3_2)
        all = [np.ndarray.flatten(hh), np.ndarray.flatten(bd), np.ndarray.flatten(sd)]


        '''
        ==== Take HD as output
        '''
        #all = [Y[:, 0], Y[:, 1], Y[:, 2]]
        np.save(savepath, all)


