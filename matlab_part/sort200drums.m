function allPath = sort200drums(allData)

allPath = cell(3, 1);

hh = {};
kd = {};
sd = {};

for i = 1:length(allData)
    filename = allData(i).name;
    filename_lower = lower(filename);
    %search for HH
    if strfind(filename_lower, 'hh')
        hh{end+1} = allData(i).path;
    elseif strfind(filename_lower, 'hat')
        hh{end+1} = allData(i).path;
    elseif strfind(filename_lower, 'hat')
        hh{end+1} = allData(i).path;
    end
    
    %search for KD
    if strfind(filename_lower, 'kick')
        kd{end+1} = allData(i).path;
    elseif strfind(filename_lower, 'kd')
        kd{end+1} = allData(i).path;
    elseif strfind(filename_lower, 'kik')
        kd{end+1} = allData(i).path;
    elseif strfind(filename_lower, 'bd')
        kd{end+1} = allData(i).path;
    end
    
    %search for SD
    if strfind(filename_lower, 'sd')
        sd{end+1} = allData(i).path;
    elseif strfind(filename_lower, 'snare')
        sd{end+1} = allData(i).path;
    elseif strfind(filename_lower, 'snr')
        sd{end+1} = allData(i).path;
    end 
end

allPath{1} = hh;
allPath{2} = kd;
allPath{3} = sd;
end