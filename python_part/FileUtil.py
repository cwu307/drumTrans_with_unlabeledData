# A collection of some of the utility functions
# CW @ GTCMT 2017

from os import listdir


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