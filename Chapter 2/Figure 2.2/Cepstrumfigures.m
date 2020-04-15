clear all
rng('default');

%%
N = 2^16;

sys = drss(10);
[h,t] = impulse(sys,N-1);
h_ceps = ifft(log(pwelch(h,[],[],'twosided')),'symmetric');
u = idinput(N,'sine')+randn([N,1]);
u_ceps = ifft(log(pwelch(u,[],[],'twosided')),'symmetric');
y_conv = conv(u,h);
y_conv_ceps = ifft(log(pwelch(y_conv,[],[],'twosided')),'symmetric');
y = lsim(sys,u,t);
y_ceps = ifft(log(pwelch(y,[],[],'twosided')),'symmetric');

%% Time domain figure

plot_length = 20;

figure
plot(u(1:plot_length+1))
hold on
plot(h(1:plot_length+1))
plot(y(1:plot_length+1))
plot(y_conv(1:plot_length+1))
title('Time domain')
xlabel('Time')
ylabel('Amplitude')

%% Cepstrum domain figure

plot_length = 20;

figure
plot(u_ceps(1:plot_length+1))
hold on
plot(h_ceps(1:plot_length+1))
plot(y_ceps(1:plot_length+1))
plot(y_conv_ceps(1:plot_length+1))
title('Cepstral domain')
xlabel('Time')
ylabel('Amplitude')

%% Save as CSV

%tempsave = [t(1:50), y_conv(1:50)];
%csvwrite('outputconv.csv',tempsave);