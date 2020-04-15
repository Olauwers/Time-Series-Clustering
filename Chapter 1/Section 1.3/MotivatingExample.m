%% Set RNG seed
rng('default');
%% Generate signals
damping = -0.05;
t = (0:0.01:11-0.001);
ys = (sin(10*t)+0.001*randn(size(t))).*exp(damping*t);
%ys = (sin(10*t)).*exp(damping*t);
yc = (cos(10*t)+0.001*randn(size(t))).*exp(damping*t);
%yc = (cos(10*t)).*exp(damping*t);

random = (0.7071*randn(size(t))).*exp(damping*t);
%% Euclidean

V = ys-yc;
distEuclid = sqrt(V*V');
Wsine = ys - random;
distEuclidSineRand = sqrt(Wsine*Wsine');
Wcosine = yc - random;
distEuclidCosineRand = sqrt(Wcosine*Wcosine');

%% Cepstral

ysceps = ifft(log(pwelch(ys,[],[],'twosided')),'symmetric');
%ysceps = ifft(log(pmtm(ys,'twosided')),'symmetric');
ycceps = ifft(log(pwelch(yc,[],[],'twosided')),'symmetric');
%ycceps = ifft(log(pmtm(yc,'twosided')),'symmetric');
randceps = ifft(log(pwelch(random,[],[],'twosided')),'symmetric');
cutoff = floor(size(ysceps,1)/2);
%cutoff = 100;
weights = 1:1:cutoff;
distCeps = weights*(ysceps(1:cutoff) - ycceps(1:cutoff)).^2;
distCepsSineRand =  weights*(ysceps(1:cutoff) - randceps(1:cutoff)).^2;
distCepsCosineRand =  weights*(ycceps(1:cutoff) - randceps(1:cutoff)).^2;

%% Cosine

distCos = dot(ys,yc)/(norm(ys,2)*norm(yc,2));
distCosSineRand = dot(ys,random)/(norm(ys,2)*norm(random,2));
distCosCosineRand = dot(yc,random)/(norm(yc,2)*norm(random,2));

%% Models

np = 1;

syssine = tf(1,polydata(ar(ys,np)),0.01);
syscosine = tf(1,polydata(ar(yc,np)),0.01);
sysrandom = tf(1,polydata(ar(random,np)),0.01);
twonormsinerand = norm(syssine-sysrandom,2);
twonormcosinerand = norm(syscosine-sysrandom,2);
twonormsinecosine = norm(syssine-syscosine,2);
infnormsinerand = norm(syssine-sysrandom,Inf);
infnormcosinerand = norm(syscosine-sysrandom,Inf);
infnormsinecosine = norm(syssine-syscosine,Inf);

%% Median, mean and standard deviation

meanSine = mean(ys);
meanCosine = mean(yc);
meanRand = mean(random);

medianSine = median(ys);
medianCosine = median(yc);
medianRand = median(random);

stdSine = std(ys);
stdCosine = std(yc);
stdRand = std(random);


%% Save as CSV

%tempsave = [t', random'];
%csvwrite('random.csv',tempsave);