clear all
rng('default')

%% Generate data

N = 2^10;    % Initialize length of signals
Ts = 10^-8;  % Initialize sampling time
inputOrder = 2;

sys1 = zpk([],[0.7 0.8],1,Ts);    
sys2 = zpk([],[0.1 0.3],1,Ts);

sys_input1 = getRandomStableSystem(inputOrder,0,0);
sys_input1.ts=Ts;
sys_input2 = getRandomStableSystem(inputOrder,0,0);
sys_input2.ts=Ts;


input1 = lsim(sys_input1,randn(N,1));
output1 = lsim(sys1, input1);    % Simulate output

input2 = lsim(sys_input2,randn(N,1));
output2 = lsim(sys2, input2);    % Simulate output

%% Calculate subspace angles using Vandermonde matrices

cascade_zeros = [pole(sys1);tzero(sys2)];
cascade_poles = [tzero(sys1);pole(sys2)];

v1 = vanderm(cascade_zeros',1000)';
v2 = vanderm(cascade_poles',1000)';

[Q1,R1] = qr(v1,0);
[Q2,R2] = qr(v2,0);

C = svd(Q1.'*Q2)

%% Calculate subspace angles using LQ on input-output data
rows = floor(N/10);

U1 = hankel(input1(1:rows),input1(rows:end));
Y1 = hankel(output1(1:rows),output1(rows:end));
U2 = hankel(input2(1:rows),input2(rows:end));
Y2 = hankel(output2(1:rows),output2(rows:end));

cosines = cosines_inputoutput(U1,Y1,U2,Y2)

%% Convergence behaviour
convergingcosines = zeros(N,4);

for l = 100:N
    rows = floor(l/10);

    U1 = hankel(input1(1:rows),input1(rows:l));
    Y1 = hankel(output1(1:rows),output1(rows:l));
    U2 = hankel(input2(1:rows),input2(rows:l));
    Y2 = hankel(output2(1:rows),output2(rows:l));
    
    convergingcosines(l,:) = cosines_inputoutput(U1,Y1,U2,Y2);
end

csvwrite('angle1.csv',[(1:N).' convergingcosines(:,3)]);
csvwrite('angle2.csv',[(1:N).' convergingcosines(:,4)]);