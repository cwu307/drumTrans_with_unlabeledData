# A collection of some of the utility functions
# CW @ GTCMT 2017

from os import listdir
import numpy as np

'''
input:
    parentFolder: string, directory to the parent folder
    ext: string, extension name of the interested files
output:
    filePathList: list, directory to the files
'''
def getFilePathList(folderpath, ext):
    allfiles = listdir(folderpath)
    filePathList = []
    for file in allfiles:
        if ext in file:
            filepath = folderpath + file
            filePathList.append(filepath)
    return filePathList

'''
input:
    data: N by 1 float vector
output:
    data_scaled: N by 1 float vector max = 1 min = 0
    maxValue: max value in original data vector
    minValue: min value in original data vector
'''
def scaleData(data):
    maxValue = max(data)
    minValue = min(data)
    data_scaled = np.zeros((len(data), 1))
    for i in range(0, len(data)):
        value = (data[i] - minValue)/ (maxValue - minValue)
        data_scaled[i] = value
    return data_scaled, maxValue, minValue
