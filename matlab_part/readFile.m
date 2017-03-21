%% Read file
% objective : read information from the assigned directory
% intput: dataDir = dir path of data
%         annDir = dir path of annotation
% output: dataInfo = data information 
%         annInfo = annotation information

function [dataInfo, annInfo, flag] = readFile(dataDir, annDir)
flag = 0; 

% read the information file 
dataInfo = recursiveFileList(dataDir, 'wav');
annInfo = recursiveFileList(annDir, 'txt');
if isempty(annInfo)
    annInfo = recursiveFileList(annDir, 'mat');
    flag = 1; %notification of reading mat
end

