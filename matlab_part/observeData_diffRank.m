%% This script is to observe the activation function from 
% multiple PFNMF
% CW @ GTCMT 2017

clear all; clc; close all;

%% add nmf drum toolbox
nmfDrumPath = '/Users/cw/Documents/CW_FILES/02_Github_repo/GTCMT/NmfDrumToolbox/src';
addpath(nmfDrumPath);

%% load target file
examplePath = '/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW_ENST_minus_one_wet_new_ratio/drummer3/audio/130_min.wav';
annPath = '/Users/cw/Documents/CW_FILES/04_Datasets/Database2/CW_ENST_minus_one_wet_new_ratio/drummer3/annotation/130_min.txt';

%==== read audio file and down-mixing + resampling
[x, fs] = audioread(examplePath);
x = mean(x, 2);
x = resample(x, 44100, fs); %sample rate consistency
fs = 44100;
fileId = fopen(annPath, 'r');
c = textscan(fileId,'%f %s');
onsets = c{1};
inst = c{2};

%==== take a short segment
x_short = x(round(10*fs):round(15*fs));
onsets_short = onsets(onsets >= 10 & onsets <= 15) - 10;
inst_short    = inst(onsets >= 10 & onsets <= 15);
onsets_short_sample = round(onsets_short * fs);
addVerticalLines(x_short, onsets_short_sample);

%==== multiple NMFs
param.hopSize = 128;

% NMF rank = 10
pfnmf_rank = 10;
param.windowSize = 2048;

load(['./templates/template_enst_', num2str(param.windowSize), '.mat'])
param.WD = template;
overlap = param.windowSize - param.hopSize;
X = spectrogram(x_short, param.windowSize, overlap, param.windowSize, fs);    
X = abs(X);
[~, HD, ~, ~, ~] = PfNmf(X, param.WD, [], [], [], pfnmf_rank, 0);
HD_scaled_rh10 = minmaxScale(HD);

% NMF rank = 50
pfnmf_rank = 50;
param.windowSize = 2048;

load(['./templates/template_enst_', num2str(param.windowSize), '.mat'])
param.WD = template;
overlap = param.windowSize - param.hopSize;
X = spectrogram(x_short, param.windowSize, overlap, param.windowSize, fs);    
X = abs(X);
[~, HD, ~, ~, ~] = PfNmf(X, param.WD, [], [], [], pfnmf_rank, 0);
HD_scaled_rh50 = minmaxScale(HD);

% NMF rank = 100
pfnmf_rank = 100;
param.windowSize = 2048;

load(['./templates/template_enst_', num2str(param.windowSize), '.mat'])
param.WD = template;
overlap = param.windowSize - param.hopSize;
X = spectrogram(x_short, param.windowSize, overlap, param.windowSize, fs);    
X = abs(X);
[~, HD, ~, ~, ~] = PfNmf(X, param.WD, [], [], [], pfnmf_rank, 0);
HD_scaled_rh100 = minmaxScale(HD);

%% visualization 
HD_scaled_rh100 = HD_scaled_rh100(:, 1:size(HD_scaled_rh100, 2));
HD_scaled_rh10 = HD_scaled_rh10(:, 1:size(HD_scaled_rh100, 2));
HD_scaled_rh50 = HD_scaled_rh50(:, 1:size(HD_scaled_rh100, 2));

inst_num = inst2num(inst_short);

% HH
onsets_hh = onsets_short(inst_num == 1);
onsets_hh_block = round(onsets_hh / (param.hopSize/fs));

figure;
plot(HD_scaled_rh100(1, :), 'r'); hold on;
plot(HD_scaled_rh10(1, :), 'g'); hold on;
plot(HD_scaled_rh50(1, :), 'b'); hold on;
stem(onsets_hh_block, ones(length(onsets_hh_block), 1), 'k');
legend('100', '10', '50');

% kd
onsets_kd = onsets_short(inst_num == 2);
onsets_kd_block = round(onsets_kd / (param.hopSize/fs));

figure;
plot(HD_scaled_rh100(2, :), 'r'); hold on;
plot(HD_scaled_rh10(2, :), 'g'); hold on;
plot(HD_scaled_rh50(2, :), 'b'); hold on;
stem(onsets_kd_block, ones(length(onsets_kd_block), 1), 'k');
legend('100', '10', '50');

% sd
onsets_sd = onsets_short(inst_num == 3);
onsets_sd_block = round(onsets_sd / (param.hopSize/fs));

figure;
plot(HD_scaled_rh100(3, :), 'r'); hold on;
plot(HD_scaled_rh10(3, :), 'g'); hold on;
plot(HD_scaled_rh50(3, :), 'b'); hold on;
stem(onsets_sd_block, ones(length(onsets_sd_block), 1), 'k');
legend('100', '10', '50');

%% remove nmf drum toolbox
rmpath(nmfDrumPath);