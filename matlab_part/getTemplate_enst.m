%% Get training template
% objective : get template data from single instrument samples
% Chih-Wei Wu, GTCMT, 2013/10

clear all; clc; close all;
profile on;

%set parameters
windowSize = 2048; 
hopSize = 512; 
w = hann(windowSize); %hann window
savepath = ['./templates/template_ENST_', num2str(windowSize), '_', num2str(hopSize), '.mat'];

%initialization
template = [];
count = 0;

%get data dir
[dataDir, annDir] = readTrainPath; %HH, KD, SD
L = length(dataDir);

tic;
for j = 1:L    
%read folder information
[dataInfo, annInfo, ~] = readFile(dataDir{j}, annDir{j});
trackNum = length(dataInfo);
subW = []; 
concateSegments = [];
    for i = 1:trackNum
        
    %============== Signal input ==============
    %load individual data name from the folder
        filename = dataInfo(i).path;
        annName = annInfo(i).path;
        fprintf('Working on data #%g...\n',i);
        fprintf('Current audio file name      = %s \n', dataInfo(i).name);
        fprintf('Current annotation file name = %s \n', annInfo(i).name);

    %load wave file
        [x, fs] = audioread(filename); 
        x = mean(x,2); %down-mixing   
        [timeAnn, drumAnn] = annRead(annName);
        
    %feature extraction 
        X = spectrogram(x, windowSize, (windowSize- hopSize), windowSize, fs);
        X_mag = abs(X);  
        
    %=========== template extraction ========== 
        loc = round(timeAnn./(hopSize/fs)); %using true locations
        numOnsets = length(loc);
        %for every onset, take more frames
        for k = 1:numOnsets 
            is = loc(k);
            ie = loc(k) + 3;
            %make sure it won't exceed the matrix range
            if (is < 1)
                is = 1;
            elseif (ie > size(X_mag, 2))
                ie = size(X_mag, 2);
            end
            segment = X_mag(:, is:ie);
            concateSegments = [concateSegments, segment]; %concatenate all the segments

        end
        
    %just to keep track on what's going on
        count = count + 1;
        if (mod(count, 5) == 0)
            fprintf('<Get Template> working...file count = %d\n',count);
        end
    end
    
    tmpW = median(concateSegments, 2); %template from one file
    subW = [subW, tmpW]; %all templates from same type of drum
    
    
%============== construct template ==============    
template = [template,subW]; %all templates from all types of drums    
end
toc;
inst = ['CHH', 'KD', 'SD'];
save(savepath, 'template', 'inst');



