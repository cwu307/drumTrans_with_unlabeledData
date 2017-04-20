'''
evaluate the transcription results on ENST dataset
CW @ GTCMT 2017
'''
import numpy as np
from FileUtil import getFilePathList
from transcriptUtil import medianThreshold, findPeaks, getIndividualOnset, showAllResults
from mir_eval.onset import f_measure
from scipy.io import loadmat

'''
==== User input
'''
targetDrummers = ['drummer1',
                'drummer2',
                'drummer3']
parentFolder     = '../../unlabeledDrumDataset/evaluation_enst/Activations/'
annotationFolder = '../../unlabeledDrumDataset/evaluation_enst/Annotations/'
savepath = '../../unlabeledDrumDataset/evaluation_enst/Evaluation_results/enst_all_results_1nn_50songs_bs64_cqt.npy'
# parentFolder     = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/Activations/'
# annotationFolder = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/Annotations/'
# savepath = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/Evaluation_results/enst_all_results.npy'

#==== define parameters
hopSize = 512.00
fs = 44100.00
order = round(0.1 / (hopSize / fs))
offset = 0.12

'''
==== File IO + testing
'''
hh_all_results = []
bd_all_results = []
sd_all_results = []

for drummer in targetDrummers:
    #==== get STFT
    activpath = parentFolder + drummer + '/'
    annpath   = annotationFolder + drummer + '/'
    activFilePathList = getFilePathList(activpath, 'npy')
    annFilePathList   = getFilePathList(annpath, 'mat')

    for i in range(0, len(activFilePathList)):
        print 'evaluate file %f' % i
        filename = activFilePathList[i]
        all = np.load(filename)
        filename_ann = annFilePathList[i]
        ann = loadmat(filename_ann)
        drums_ann = ann['drums']
        onsets_ann = ann['onsets']
        ref_hh, ref_bd, ref_sd = getIndividualOnset(onsets_ann, drums_ann)
        #==== load activation functions
        activ_hh = all[0]
        activ_bd = all[1]
        activ_sd = all[2]

        #==== onset detection with adaptive threshold
        thresCurve_hh = medianThreshold(activ_hh, order, offset)
        thresCurve_bd = medianThreshold(activ_bd, order, offset)
        thresCurve_sd = medianThreshold(activ_sd, order, offset)

        dump, onsetsInSec_hh = findPeaks(activ_hh, thresCurve_hh, fs, hopSize)
        dump, onsetsInSec_bd = findPeaks(activ_bd, thresCurve_bd, fs, hopSize)
        dump, onsetsInSec_sd = findPeaks(activ_sd, thresCurve_sd, fs, hopSize)

        #==== calculate (F-measure, Precision, Recall)
        hh_result = f_measure(ref_hh, onsetsInSec_hh, window=0.05)
        bd_result = f_measure(ref_bd, onsetsInSec_bd, window=0.05)
        sd_result = f_measure(ref_sd, onsetsInSec_sd, window=0.05)

        print 'hh_results\n'
        print hh_result
        print 'bd_results\n'
        print bd_result
        print 'sd_results\n'
        print sd_result

        #==== keep all results
        hh_all_results.append(hh_result)
        bd_all_results.append(bd_result)
        sd_all_results.append(sd_result)

    #==== save results per drummer
    all_results = [hh_all_results, bd_all_results, sd_all_results]
    np.save(savepath, all_results)

showAllResults(savepath)