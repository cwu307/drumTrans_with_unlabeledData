%% This script is to extract SFTF from evaluation set
% CW @ GTCMT 2017

clear all; clc; close all;

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
              
tic;
for g = 1:length(selectedDrummer)
    subpath = [datasetPath, selectedDrummer{g}, '/', 'audio'];
    subpath_ann = [datasetPath, selectedDrummer{g}, '/', 'annotation'];
    subdata = recursiveFileList(subpath, 'wav');
    subdata_ann = recursiveFileList(subpath_ann, 'txt');
    
    %==== create dir
    savepath_stft = [saveFolderPath, 'STFT/', selectedDrummer{g}];
    savepath_ann  = [saveFolderPath, 'Annotations/', selectedDrummer{g}];
    mkdir(savepath_stft);
    mkdir(savepath_ann);
    
    for i = 1:length(subdata) %only take the first 10 songs
        fprintf('Processing drummer %g, song %g\n', g, i);
        %==== read audio file and down-mixing + resampling
        [x, fs] = audioread(subdata(i).path);
        x = mean(x, 2);
        x = resample(x, 44100, fs); %sample rate consistency
        fs = 44100;
        X = spectrogram(x, param.windowSize, overlap, param.windowSize, fs); 
        X = abs(X);
        
        %==== Annotation
        [onsets, drums] = annRead(subdata_ann(i).path);
        save([savepath_ann, '/', subdata(i).name(1:end-3), 'mat'], 'onsets', 'drums');
        
        %==== STFT
        save([savepath_stft, '/', subdata(i).name(1:end-3), 'mat'], 'X');
    end
end
toc;
