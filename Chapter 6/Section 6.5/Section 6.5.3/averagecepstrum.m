%% Initialization of the workspace:
close all
clearvars

%% Initial test on synthetic data:
% Generate a test model:
sys1 = zpk([],[0.5,0.5j,-0.5j],1,1); % Third order AR model.

poles1 = pole(sys1); % Poles: 0.5, 0.5j, and -0.5j
transf1 = tf(sys1); % TF: a(z) = z^3 - 0.5 z^2 + 0.25 z - 0.125

sys2 = zpk([],[0.4,0.55j,-0.55j],1,1); % Third order AR model.

poles2 = pole(sys2); 
transf2 = tf(sys2); 

% Generate the random input/output-data:
N = 2^15;
t = 0:1:N-1;
samples = 100;
input1 = randn(samples,size(t,2));
output1 = zeros(size(input1));
for i = 1:size(input1,1)
    output1(i,:) = lsim(sys1,input1(i,:));
end
input2 = randn(samples,size(t,2));
output2 = zeros(size(input1));
for i = 1:size(input2,1)
    output2(i,:) = lsim(sys2,input2(i,:));
end
figure()
plot(output1(1,1:200))
hold
plot(output2(1,1:200))

% Calculate the poles:
ceps1 = ifft(log(pwelch(output1(1,:),[],[],'twosided')),'symmetric');
sysceps1 = tf(1,cepsarid(ceps1,3)',-1);
polesceps1 = pole(sysceps1);

ceps2 = ifft(log(pwelch(output2(1,:),[],[],'twosided')),'symmetric');
sysceps2 = tf(1,cepsarid(ceps2,3)',-1);
polesceps2 = pole(sysceps2);



%% Cepstral clustering

disp('Start Original Cepstral Distance')
tic
DistCepstral = zeros(2*samples,2*samples);

outputs = [output1;output2];

OutputCeps = zeros(2*samples,size(ifft(log(pwelch(outputs(1,:),[],[],'twosided')),'symmetric'),1));
%cepstrumcutoff = floor(size(ifft(log(pwelch(outputs(1,:),[],[],'twosided')),'symmetric'),1));
cepstrumcutoff = 50;
weights = 1:1:cepstrumcutoff-1;
for i = 1:2*samples
    OutputCeps(i,:) = ifft(log(pwelch(outputs(i,:),[],[],'twosided')),'symmetric');
end

clear i

WeightedCeps = zeros(2*samples,cepstrumcutoff-1);

for i = 1:2*samples
    WeightedCeps(i,:) = sqrt(weights).*OutputCeps(i,2:cepstrumcutoff);
end
idx = kmeans(WeightedCeps,2);

toc

%% Average cepstra

cepsmodels = zeros(2*samples,4);

for i = 1:2*samples
    cepsmodels(i,:) = cepsarid(OutputCeps(i,:),3);
end

w = logspace(-2,pi,1000);
responses = zeros(2*samples,length(w));

for i = 1:2*samples
    responses(i,:) = squeeze(freqresp(tf(1,cepsmodels(i,:),1),w));
    disp(i)
end

responseaverage1 = geomean(abs(responses(1:samples,:)).^2,1);
euclideanaverage1 = mean(abs(responses(1:samples,:)).^2,1);
reseponseaverage2 = geomean(abs(responses(1+samples:2*samples,:)).^2,1);
euclideanaverage2 = mean(abs(responses(1+samples:2*samples,:)).^2,1);

averageceps1 = mean(OutputCeps(1:samples,:),1);
averagemodel1 = cepsarid(averageceps1,3);
sysaveragemodel1 = tf(1,averagemodel1',1);
averageresponse1 = abs(squeeze(freqresp(sysaveragemodel1,w))).^2;

averageceps2 = mean(OutputCeps(1+samples:2*samples,:),1);
averagemodel2 = cepsarid(averageceps2,3);
sysaveragemodel2 = tf(1,averagemodel2',1);
averageresponse2 = abs(squeeze(freqresp(sysaveragemodel2,w))).^2;

cepsofeuclideanaverage1 = ifft(log(pwelch(mean(output1,1),[],[],'twosided')),'symmetric');
euclideanaveragemodel1 = cepsarid(cepsofeuclideanaverage1,3);
syseuclideanaveragemodel1 = tf(1,euclideanaveragemodel1',1);
euclideanaverageresponse1 = abs(squeeze(freqresp(syseuclideanaveragemodel1,w))).^2;

figure()
semilogx(w,mag2db(abs(squeeze(freqresp(sys1,w))).^2),'black','DisplayName','Actual')
hold
plot(w,mag2db(responseaverage1),'DisplayName','Geometric average')
plot(w,mag2db(averageresponse1),'--','DisplayName','Cluster center')
%plot(w,mag2db(euclideanaverage1),':')
%plot(w,mag2db(abs(squeeze(responses(1,:))).^2))
plot(w,mag2db(euclideanaverageresponse1),'DisplayName','Euclidean average')
ylim([-2 8.5])
legend('Location','southwest')

%%

% csvwrite('ccepstral.csv',[((1:2*samples) - 1)', ccepstral]);
% csvwrite('output1.csv',[((1:size(output1,2))-1)', output1(1,:)']);
% csvwrite('output2.csv',[((1:size(output2,2))-1)', output2(1,:)']);
% csvwrite('truepsd.csv',[w',mag2db(abs(squeeze(freqresp(sys1,w))).^2)]);
% csvwrite('responseofcepstralcenter.csv',[w',mag2db(responseaverage1')]);
% csvwrite('averageofresponses.csv',[w',mag2db(averageresponse1)]);
% csvwrite('responseofeuclideanaverage.csv',[w',mag2db(euclideanaverageresponse1)]);


%%
% csvwrite('sys1response1.csv',[w', mag2db(abs(squeeze(responses(1,:))).^2)']);
% csvwrite('sys1response2.csv',[w', mag2db(abs(squeeze(responses(2,:))).^2)']);
% csvwrite('sys1response3.csv',[w', mag2db(abs(squeeze(responses(3,:))).^2)']);
% csvwrite('sys1response4.csv',[w', mag2db(abs(squeeze(responses(4,:))).^2)']);
% csvwrite('sys1response5.csv',[w', mag2db(abs(squeeze(responses(5,:))).^2)']);
% csvwrite('sys2response1.csv',[w', mag2db(abs(squeeze(responses(101,:))).^2)']);
% csvwrite('sys2response2.csv',[w', mag2db(abs(squeeze(responses(102,:))).^2)']);
% csvwrite('sys2response3.csv',[w', mag2db(abs(squeeze(responses(103,:))).^2)']);
% csvwrite('sys2response4.csv',[w', mag2db(abs(squeeze(responses(104,:))).^2)']);
% csvwrite('sys2response5.csv',[w', mag2db(abs(squeeze(responses(105,:))).^2)']);





