%% calculate the median adaptive threshold
% input: nvt = m*n novelty function
%        b = user defined 1*j window , j = order
%        lamda = a variable for controlling the sensitivity (% of max)
% output:Gdma = m*n adaptive threshold 

function [Gdme] = myMedianThres(nvt,K, lamda)

[m, n] = size(nvt);
maxVal = max(nvt, [], 2);

for i = 1:n

    if i-K < 1            
        Gdme(:,i) = nvt(:,1) + lamda*maxVal;            
    elseif i-K >= 1
        med = median(nvt(:,(i-K):i), 2);
        Gdme(:,i) = lamda*maxVal + med;
    end

end




%compensate the delay of the threshold 

shiftSize = round(0.5 * K); %1/2 order size

Gdme(:, 1:(end-shiftSize)) = Gdme(:, (shiftSize+1):end);