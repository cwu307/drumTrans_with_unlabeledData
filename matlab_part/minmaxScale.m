% scale vector to range between 0 and 1
% input:
%   x = m by n float matrix, m activation function with length of n 
% output:
%   x_scaled = 1 by N vector with numerical range of {0, 1}
% CW @ GTCMT 2017

function x_scaled = minmaxScale(x)

[m, n] = size(x);
assert(m <= n, 'check the dimensionality again');
x_scaled = zeros(m, n);

for i = 1:m
    nvt = x(i, :);
    minVal = min(nvt);
    maxVal = max(nvt);
    x_scaled(i, :) = (nvt - minVal)./(maxVal - minVal);
end

end