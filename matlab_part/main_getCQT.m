%% This script is to extract SFTF from evaluation set
% CW @ GTCMT 2017

clear all; clc; close all;
addpath('/Users/cw/Documents/CW_FILES/03_Toolboxes/Matlab programs/labROSA/');


%% load target file
datasetPath = '/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW_ENST_minus_one_wet_new_ratio/';
saveFolderPath = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/';
selectedDrummer = {'drummer1';
                  'drummer2';
                  'drummer3'};
              
%% define parameters        
param.windowSize = 2048;
param.hopSize = 512;
overlap = param.windowSize - param.hopSize;  
fmin = 20;
fmax = 20000;
bins = 12;
              
tic;
for g = 1:length(selectedDrummer)
    subpath = [datasetPath, selectedDrummer{g}, '/', 'audio'];
    subdata = recursiveFileList(subpath, 'wav');
    
    %==== create dir
    savepath_cqt = [saveFolderPath, 'CQT/', selectedDrummer{g}];
    mkdir(savepath_cqt);
    
    for i = 1:length(subdata) %only take the first 10 songs
        fprintf('Processing drummer %g, song %g\n', g, i);
        %==== read audio file and down-mixing + resampling
        [x, fs] = audioread(subdata(i).path);
        x = mean(x, 2);
        x = resample(x, 44100, fs); %sample rate consistency
        fs = 44100;
        
        %==== using Dan Ellis's code to perform the mapping
        [Xcqt, mx] = logfsgram(x, param.windowSize, fs, param.windowSize, overlap, fmin, bins);

        %==== CQT
        save([savepath_cqt, '/', subdata(i).name(1:end-3), 'mat'], 'Xcqt');
    end
end
toc;

rmpath('/Users/cw/Documents/CW_FILES/03_Toolboxes/Matlab programs/labROSA/');
