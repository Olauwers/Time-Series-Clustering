clear all
rng('default')

%% Initialize parameters

inputOrder = 5;
NFFT = 2^11;
cutoff = 100;

%% Generate Systems and Input-Output Pairs

N = 2^16;    % Initialize length of signals
Ts = 10^-8;


sys1 = zpk([0.5,0.7+0.1i,0.7-0.1i,0.5 + 0.6i,0.5 - 0.6i],[0.7,0.6,0.3,0.6+0.2i,0.6-0.2i],1,1);    
sys1.ts = Ts;

sys2 = zpk([0.75,0.7+0.3i,0.7-0.3i,0.2 + 0.8i,0.2 - 0.8i],[0.95,0.55,0.15,0.9+0.1i,0.9-0.1i],1,1);
sys2.ts = Ts;

order = size(pole(sys1,1)); % Note the order of the system


syscasc = sys1*sys2^-1;

sys_input1 = getRandomStableSystem(inputOrder,inputOrder,0);
sys_input1.ts=Ts;
input1 = lsim(sys_input1,randn(N,1));
output1 = lsim(sys1, input1);    % Simulate output

sys_input2 = getRandomStableSystem(inputOrder,inputOrder,0);
sys_input2.ts=Ts;
input2 = lsim(sys_input2,randn(N,1));
output2 = lsim(sys2, input2);    % Simulate output


%% Calculate Subspace Angles

cascade_zeros = [pole(sys1);tzero(sys2)];
cascade_poles = [tzero(sys1);pole(sys2)];

v1 = vanderm(cascade_zeros',1000)';
v2 = vanderm(cascade_poles',1000)';

[Q1,R1] = qr(v1,0);
[Q2,R2] = qr(v2,0);

C = svd(Q1.'*Q2);
distance_subspace_angles = -2*sum(log(abs(C)));

%% Calculate Poles and Zeroes Distance

distance_poles_and_zeroes = sum(sum(log((1-cascade_poles*cascade_zeros').^2))) - sum(sum(log(1-cascade_poles*cascade_poles'))) - sum(sum(log(1-cascade_zeros*cascade_zeros')));

%% Calculate Cepstrum Distance

weights = 0:cutoff-1;

input1_cepstrum = ifft(log(pwelch(input1,[],[],NFFT,1/Ts,'twosided')),'symmetric');
output1_cepstrum = ifft(log(pwelch(output1,[],[],NFFT,1/Ts,'twosided')),'symmetric');

input2_cepstrum = ifft(log(pwelch(input2,[],[],NFFT,1/Ts,'twosided')),'symmetric');
output2_cepstrum = ifft(log(pwelch(output2,[],[],NFFT,1/Ts,'twosided')),'symmetric');

distance_weighted_cepstral = weights*((output1_cepstrum(1:cutoff) - input1_cepstrum(1:cutoff))-(output2_cepstrum(1:cutoff) - input2_cepstrum(1:cutoff))).^2;


%% Generate pdf's of weighted cepstral distance

% repetitions = 10000;
% 
% %Change parameter values if wanted
% NFFT = 2^11;
% cutoff = 100;
% inputOrder = 1;
% 
% weightedCepstralDistancesRepetitions = zeros(repetitions,1);
% tic
% for i = 1:repetitions
%     weightedCepstralDistancesRepetitions(i) = deterministicWeightedCepstralDistance(sys1,sys2,N,Ts,inputOrder,NFFT,cutoff);
%     if mod(i,floor(repetitions/20)) == 0 
%         disp(i)
%     end
% end
% toc
% disp(['Finished with NFFT = ', num2str(NFFT), ', cutoff = ', num2str(cutoff), ', inputorder = ', num2str(inputOrder)])
% figure
% ksdensity(weightedCepstralDistancesRepetitions,linspace(5,8,1000))
% %figure
% %histogram(weightedCepstralDistancesRepetitions,'Normalization','pdf','displaystyle','stairs','binmethod','fd')

%% Calculate Kernel distance using QR
rows = 200;

U1 = hankel(input1(1:rows),input1(rows:end));
Y1 = hankel(output1(1:rows),output1(rows:end));
U2 = hankel(input2(1:rows),input2(rows:end));
Y2 = hankel(output2(1:rows),output2(rows:end));

cosines = cosines_inputoutput(U1,Y1,U2,Y2);
distance_kernel = -2*sum(log(abs(cosines)));

%% Save CSV's

% F = ksdensity(weightedCepstralDistancesRepetitions,linspace(5,8,1000));
% csvwrite('Nfft2to11cutoff200inputorder20pdf.csv',[linspace(5,8,1000)' F'])
% csvwrite('Nfft2to11cutoff200inputorder20distances.csv',weightedCepstralDistancesRepetitions)

%%