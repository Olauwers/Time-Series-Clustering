%% Initialization of the workspace:
close all
clearvars

%% Initial test on synthetic data:
% Generate a test model:
sys = zpk([],[0.5,0.5j,-0.5j],1,1); % Third order AR model.

poles = pole(sys); % Poles: 0.5, 0.5j, and -0.5j
transf = tf(sys); % TF: a(z) = z^3 - 0.5 z^2 + 0.25 z - 0.125

% Generate the random input/output-data:
N = 5*10^5;
t = 0:1:N-1;
input = randn(size(t));
output = lsim(sys,input);

% Calculate the poles:
ceps = ifft(log(pwelch(output,[],[],'twosided')),'symmetric');
sysceps = tf(1,cepsarid(ceps,3)',-1);
polesceps = pole(sysceps);

% Visualize the results:
figure(1)
clf
hold on
pzmap(sys)
plot(real(polesceps),imag(polesceps),'*')
hold off
title('Poles')
legend('original','cepstrum')

[mag1,~,wout1] = bode(sys);
[mag2,~,wout2] = bode(sysceps);

figure(2)
clf
hold on
plot(wout1,20*log10(squeeze(mag1)))
plot(wout2,20*log10(squeeze(mag2)))
hold off
title('Bode diagram')
legend('original','cepstrum')

%% Convergence test on synthetic data:
% Compare for different data lengths:
datarange = zeros(31,1);
eceps = zeros(31,100);
els = zeros(31,100);
for k = 10:40
    N = 2^(k/2);
    datarange(k-9) = N;
    t = 0:1:N-1;
    for l = 1:100
        input = randn(size(t));
        output = lsim(sys,input);
        ceps = ifft(log(pwelch(output,[],[],'twosided')),'symmetric');
        eceps(k-9,l) = norm(tf(1,cepsarid(ceps,3)',-1) - sys);
        els(k-9,l) = norm(tf(1,[1; getpvec(ar(output,3,'ls'))]',-1) - sys);
    end
end
stdceps = std(eceps,0,2);
meanceps = mean(eceps,2);
stdls = std(els,0,2);
meanls = mean(els,2);

% Visualize the results:
figure(3)
clf
semilogx(datarange,[meanceps meanls meanceps+stdceps meanceps-stdceps meanls+stdls meanls-stdls])
title('Convergence of both algorithms')
legend('cepstrum','least')
xlabel('N')
ylabel('H2-norm of the error system')

%% Noise test on synthetic data:
% Generate the random input/output-data:
N = 2^20;
t = 0:1:N-1;
input = randn(size(t));
output = lsim(sys,input);

% Calculate the poles for different SNRs:
output1 = output + 0*randn(size(t))'; % Inf
ceps1 = ifft(log(pwelch(output1,[],[],'twosided')),'symmetric');
sysceps1 = tf(1,cepsarid(ceps1,3)',-1);

output2 = output + 0.01*randn(size(t))'; % 40 dB
ceps2 = ifft(log(pwelch(output2,[],[],'twosided')),'symmetric');
sysceps2 = tf(1,cepsarid(ceps2,3)',-1);

output3 = output + 0.1*randn(size(t))'; % 20 dB
ceps3 = ifft(log(pwelch(output3,[],[],'twosided')),'symmetric');
sysceps3 = tf(1,cepsarid(ceps3,3)',-1);

output4 = output + 0.30*randn(size(t))'; % 10.4576 dB
ceps4 = ifft(log(pwelch(output4,[],[],'twosided')),'symmetric');
sysceps4 = tf(1,cepsarid(ceps4,3)',-1);

output5 = output + 0.5*randn(size(t))'; % 6.0206 dB
ceps5 = ifft(log(pwelch(output5,[],[],'twosided')),'symmetric');
sysceps5 = tf(1,cepsarid(ceps5,3)',-1);

output6 = output + 0.891*randn(size(t))'; % 1.0024 dB
ceps6 = ifft(log(pwelch(output6,[],[],'twosided')),'symmetric');
sysceps6 = tf(1,cepsarid(ceps6,3)',-1);

% Visualize the results:
[mag1,~,wout1] = bode(sys);
[mag2,~,wout2] = bode(sysceps1);
[mag3,~,wout3] = bode(sysceps2);
[mag4,~,wout4] = bode(sysceps3);
[mag5,~,wout5] = bode(sysceps4);
[mag6,~,wout6] = bode(sysceps5);
[mag7,~,wout7] = bode(sysceps6);
figure(4)
clf
hold on
plot(wout1,20*log10(squeeze(mag1)))
plot(wout2,20*log10(squeeze(mag2)))
plot(wout3,20*log10(squeeze(mag3)))
plot(wout4,20*log10(squeeze(mag4)))
plot(wout5,20*log10(squeeze(mag5)))
plot(wout6,20*log10(squeeze(mag6)))
plot(wout7,20*log10(squeeze(mag7)))
hold off
title('Bode diagram')
legend('original','Inf','40dB','20dB','10db','6dB','1dB')


