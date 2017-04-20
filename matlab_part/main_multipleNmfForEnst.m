%% This script is to observe the activation function from 
% multiple PFNMF
% CW @ GTCMT 2017

clear all; clc; close all;

%% add nmf drum toolbox
nmfDrumPath = '/Users/cw/Documents/CW_FILES/02_Github_repo/GTCMT/NmfDrumToolbox/src';
addpath(nmfDrumPath);

%% load target file
datasetPath = '/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW_ENST_minus_one_wet_new_ratio/';
saveFolderPath = '/Volumes/CW_MBP15/Datasets/unlabeledDrumDataset/evaluation_enst/';
selectedDrummer = {'drummer1';
                  'drummer2';
                  'drummer3'};
              
%% define parameters        
pfnmf_rank = 50;
param.windowSize = 2048;
param.hopSize = 512;
overlap = param.windowSize - param.hopSize;   
              
              
%% load templates
load(['./templates/template_ENST_', num2str(param.windowSize), '_', num2str(param.hopSize), '.mat'])
template_enst = template;
load(['./templates/template_SMT_', num2str(param.windowSize), '_', num2str(param.hopSize), '.mat'])
template_smt = template;
load(['./templates/template_200DRUMS_', num2str(param.windowSize), '_', num2str(param.hopSize), '.mat'])
template_200drums = template;

 

tic;
for g = 1:length(selectedDrummer)
    subpath = [datasetPath, selectedDrummer{g}, '/', 'audio'];
    subdata = recursiveFileList(subpath, 'wav');
    
    %==== create dir
    savepath_enst = [saveFolderPath, 'enst/', selectedDrummer{g}];
    mkdir(savepath_enst);
    savepath_smt = [saveFolderPath, 'smt/', selectedDrummer{g}];
    mkdir(savepath_smt);
    savepath_200drums = [saveFolderPath, '200drums/', selectedDrummer{g}];
    mkdir(savepath_200drums);
    
    for i = 1:length(subdata) %only take the first 100 songs
        fprintf('Processing genre %g, song %g\n', g, i);
        %==== read audio file and down-mixing + resampling
        [x, fs] = audioread(subdata(i).path);
        x = mean(x, 2);
        x = resample(x, 44100, fs); %sample rate consistency
        fs = 44100;
        X = spectrogram(x, param.windowSize, overlap, param.windowSize, fs); 
        X = abs(X);
        
        %==== NMF ENST
        param.WD = template_enst;    
        [~, HD, ~, ~, ~] = PfNmf(X, param.WD, [], [], [], pfnmf_rank, 0);
        save([savepath_enst, '/', subdata(i).name(1:end-3), 'mat'], 'HD');

        %==== NMF SMT
        param.WD = template_smt;
        [~, HD, ~, ~, ~] = PfNmf(X, param.WD, [], [], [], pfnmf_rank, 0);
        save([savepath_smt, '/', subdata(i).name(1:end-3), 'mat'], 'HD');

        %==== NMF 200DRUMS
        param.WD = template_200drums;
        [~, HD, ~, ~, ~] = PfNmf(X, param.WD, [], [], [], pfnmf_rank, 0);
        save([savepath_200drums, '/', subdata(i).name(1:end-3), 'mat'], 'HD');
        
    end
end
toc;

%% remove nmf drum toolbox
rmpath(nmfDrumPath);