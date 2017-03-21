%% Read test data path
% objective : set the folder and read the list
% output: dataDir = dir path of data
%         annDir = dir path of annotation

% set the directory of data
function [dataDir, annDir] = readTrainPath


dataDir = {
'/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW test drum sounds/1CHH';
'/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW test drum sounds/3kick';
'/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW test drum sounds/4snare';
    };

annDir = {
'/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW test drum sounds/1CHH';
'/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW test drum sounds/3kick';
'/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW test drum sounds/4snare';
    };



