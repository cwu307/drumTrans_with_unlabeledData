

% visualize HH
subplot(311)
load ./templates/template_ENST_2048_512.mat
plot(template(:, 1), 'r'); hold on;
load ./templates/template_SMT_2048_512.mat
plot(template(:, 1), 'g'); hold on;
load ./templates/template_200DRUMS_2048_512.mat
plot(template(:, 1), 'b');
title('HH');
legend('ENST', 'SMT', '200');

% visualize KD
subplot(312)
load ./templates/template_ENST_2048_512.mat
plot(template(:, 2), 'r'); hold on;
load ./templates/template_SMT_2048_512.mat
plot(template(:, 2), 'g'); hold on;
load ./templates/template_200DRUMS_2048_512.mat
plot(template(:, 2), 'b');
title('KD');
legend('ENST', 'SMT', '200');

% visualize SD
subplot(313)
load ./templates/template_ENST_2048_512.mat
plot(template(:, 3), 'r'); hold on;
load ./templates/template_SMT_2048_512.mat
plot(template(:, 3), 'g'); hold on;
load ./templates/template_200DRUMS_2048_512.mat
plot(template(:, 3), 'b');
title('SD');
legend('ENST', 'SMT', '200');