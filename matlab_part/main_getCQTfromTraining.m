%% This script is to observe the activation function from 
% multiple PFNMF
% CW @ GTCMT 2017

clear all; clc; close all;
addpath('/Users/cw/Documents/CW_FILES/03_Toolboxes/Matlab programs/labROSA/');


%% load target file
datasetPath = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/audio/';
saveFolderPath = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/activations/';
selectedGenres = {'dance-club-play-songs';
                  'hot-mainstream-rock-tracks'
                  'latin-songs';
                  'pop-songs';
                  'r-b-hip-hop-songs'};
              
%% define parameters        
param.windowSize = 2048;
param.hopSize = 512;
overlap = param.windowSize - param.hopSize;   
fmin = 20;
fmax = 20000;
bins = 12; 

tic;
for g = 1:length(selectedGenres)
    subpath = [datasetPath, selectedGenres{g}];
    subdata = recursiveFileList(subpath, 'mp3');
    
    %==== create dir
    savepath_cqt = [saveFolderPath, 'CQT/', selectedGenres{g}];
    mkdir(savepath_cqt);
    
    for i = 1:30 %only take the first 100 songs
        fprintf('Processing genre %g, song %g\n', g, i);
        %==== read audio file and down-mixing + resampling
        [x, fs] = audioread(subdata(i).path);
        x = mean(x, 2);
        x = resample(x, 44100, fs); %sample rate consistency
        fs = 44100;

        %==== using Dan Ellis's code to perform the mapping
        [Xcqt, mx] = logfsgram(x, param.windowSize, fs, param.windowSize, overlap, fmin, bins);
       
        %==== STFT
        save([savepath_cqt, '/', subdata(i).name(1:end-3), 'mat'], 'Xcqt');
    end
end
toc;

rmpath('/Users/cw/Documents/CW_FILES/03_Toolboxes/Matlab programs/labROSA/');
