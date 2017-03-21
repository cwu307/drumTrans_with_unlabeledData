function inst_num = inst2num(inst)

inst_num = zeros(size(inst));
for i = 1:length(inst)
    if strcmp(inst{i}, 'chh')
        inst_num(i) = 1;
    elseif strcmp(inst{i}, 'ohh')
        inst_num(i) = 1;
    elseif strcmp(inst{i}, 'bd')
        inst_num(i) = 2;
    elseif strcmp(inst{i}, 'sd')
        inst_num(i) = 3;
    else
        inst_num(i) = 0;
    end
end

end