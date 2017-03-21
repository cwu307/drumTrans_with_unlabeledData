%% Annotation read
% objective : get onset annotation from input txt file 
% input : filename 
%         flag = read .mat or not?
% output: annotation

function [timeAnn, drumAnn] = annRead(annName)


    %read the onset time and drum type
    [time, drumAnn] = textread(annName,'%s%s','headerlines',0);

    N = length(time);
    timeAnn = zeros(N, 1);
    for i = 1:N
        timeAnn(i,1) = str2num(time{i});
    end
