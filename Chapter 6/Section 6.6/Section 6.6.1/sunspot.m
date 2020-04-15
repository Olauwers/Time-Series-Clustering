% Initialize the workspace:
% close all
clearvars
load('sunspot.mat')
%set(0,'DefaultFigureWindowStyle','docked')

p = 2;

% Generate cepstrum coefficients:
% w = detrend(w); % Because the papers also detrend the data!
ceps = ifft(log(pmtm(w,[],[],'twosided')),'symmetric');

% Identify the models via cepstrum and least-squares:
syscepsno = tf(1,cepsarid(ceps,p)',-1);
syslsno = tf(1,[1; getpvec(ar(w,p,'ls'))]',-1);

% Add the models form papers:
sysyule = tf(1,[1 -1.34254 0.65504],-1); % G. U. Yule 1927
v = w;
w = detrend(w); % Because the papers also detrend the data!
ceps = ifft(log(pmtm(w,[],[],'twosided')),'symmetric');

% Identify the models via cepstrum and least-squares:
sysceps = tf(1,cepsarid(ceps,p)',-1);
sysls = tf(1,[1; getpvec(ar(w,p,'ls'))]',-1);

% Visualize the results:
[magceps,~,woutceps] = bode(sysceps);
[magls,~,woutls] = bode(sysls);
[magyule,~,woutyule] = bode(sysyule);
figure(1)
clf
hold on
plot(woutceps,20*log10(squeeze(magceps)))
plot(woutls,20*log10(squeeze(magls)))
plot(woutyule,20*log10(squeeze(magyule)))
hold off
title('Bode diagram')
legend('cepstrum','leastsq','Yule')

% Visualize the results:
[magcepsno,~,woutcepsno] = bode(syscepsno);
[maglsno,~,woutlsno] = bode(syslsno);
figure(2)
clf
hold on
plot(woutceps,20*log10(squeeze(magceps)))
plot(woutls,20*log10(squeeze(magls)))
plot(woutcepsno,20*log10(squeeze(magcepsno)))
plot(woutlsno,20*log10(squeeze(maglsno)))
hold off
title('Bode diagram')
legend('cepstrum','leastsq','cepstrum no detrend', 'leastsq no detrend')


