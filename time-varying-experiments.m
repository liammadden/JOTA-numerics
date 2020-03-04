%% Time-varying experiments
% This script generates Figures 2a, 2b, 3a, 3b, 4, 5a, and 5b. Notation
% follows that of the manuscript. OGD, ONM, OLNM, and ORGD denote
% online gradient descent, online Nesterov's method, online long-step
% Nesterov's method, and online regularized gradient descent respectively.
% ite, fcn, and gra denote the iterate, function, and gradient errors
% respectively.

%% Clear
clear all
close all
clc

%% Figure 2a
d = 1e3;
niter = 5e2;
L = 5e2;
mu = 1;
a = (L+mu)/2;
kappa = L/mu;
alpha = 2/(mu+L);
gamma = (sqrt(kappa)-1)/(sqrt(kappa)+1);
beta = gamma;
sigma = 1;
offset = sigma*sqrt((1+gamma)/(1-gamma)); %gamma^c
lb = (sqrt(kappa)-1)/2*sigma; %lower bound

x0 = zeros(d,1);
x_OGD = x0;
x_ONM = x0;
y_ONM = x0;
x_OLNM = x0;
y_OLNM = x0;
z_OLNM = x0;

ite_OGD = zeros(niter,1);
ite_ONM = ite_OGD;
ite_OLNM = ite_OGD;

T = floor((2+sqrt(2))*sqrt(kappa));

for i = 1:niter
    disp(i)
    
    t = i-1;
    A = [a*diag(ones(t,1)) zeros(t,d-t); zeros(d-t,t) (L-mu)/4*gallery('tridiag',d-t)+mu*diag(ones(d-t,1))];
    b = offset*[a*ones(t,1); (L-mu)/4; zeros(d-t-1,1)];
    f = @(x) x'*A*x/2-x'*b;
    grad = @(x) A*x-b;
    xopt = A\b;
    ite_OGD(i) = norm(x_OGD-xopt);
    ite_ONM(i) = norm(x_ONM-xopt);
    ite_OLNM(i) = norm(x_OLNM-xopt);
    
    x_OGD = x_OGD - alpha*grad(x_OGD);
    
    if mod(i-1,T) == 0
        a = 1;
        oldgrad = grad;
        x_OLNM = z_OLNM;
        y_OLNM = z_OLNM;
    end
    na = (1+sqrt(1+4*a^2))/2;
    temp = y_OLNM-oldgrad(y_OLNM)/L;
    y_OLNM = temp + (a-1)/na*(temp-z_OLNM);
    z_OLNM = temp;
    a = na;
    
    temp = y_ONM-grad(y_ONM)/L;
    y_ONM = temp+beta*(temp-x_ONM);
    x_ONM = temp;
end

marker = linspace(niter/5,niter,5);
figure
plot(1:niter,lb*ones(niter,1),'--k',1:niter,ite_OGD,'ok-',1:niter,ite_ONM,'^r-',1:niter,ite_OLNM,'dg-','MarkerSize',10,'MarkerIndices',marker)
xlabel('Number of iterations')
ylabel('Iterate error')
lgd = legend('lower bound','online gradient descent','online Nesterov`s method','OLNM');
lgd.Location = 'northwest';

%% Figure 2b
niter = 5e2;
track_idx = 4e2:niter;
d = 1e3;
L = 5e2;
sigma = 1;
l = 10;
kappas = linspace(1e2,1e3,l);
track_OGD = zeros(l,1);
track_ONM = track_OGD;
track_OLNM = track_OGD;
for m = 1:l
kappa = kappas(m);
mu = L/kappa;
a = (L+mu)/2;
alpha = 2/(mu+L);
gamma = (sqrt(kappa)-1)/(sqrt(kappa)+1);
beta = gamma;
offset = sigma*sqrt((1+gamma)/(1-gamma));

x0 = zeros(d,1);
x_OGD = x0;
x_ONM = x0;
y_ONM = x0;
x_OLNM = x0;
y_OLNM = x0;
z_OLNM = x0;

ite_OGD = zeros(niter,1);
ite_ONM = ite_OGD;
ite_OLNM = ite_OGD;

T = floor((2+sqrt(2))*sqrt(kappa));

for i = 1:niter
    disp([m i])
    
    t = i-1;
    A = [a*diag(ones(t,1)) zeros(t,d-t); zeros(d-t,t) (L-mu)/4*gallery('tridiag',d-t)+mu*diag(ones(d-t,1))];
    b = offset*[a*ones(t,1); (L-mu)/4; zeros(d-t-1,1)];
    f = @(x) x'*A*x/2-x'*b;
    grad = @(x) A*x-b;
    xopt = A\b;
    ite_OGD(i) = norm(x_OGD-xopt);
    ite_ONM(i) = norm(x_ONM-xopt);
    ite_OLNM(i) = norm(x_OLNM-xopt);
    
    x_OGD = x_OGD - alpha*grad(x_OGD);
    
    if mod(i-1,T) == 0
        a = 1;
        oldgrad = grad;
        x_OLNM = z_OLNM;
        y_OLNM = z_OLNM;
    end
    na = (1+sqrt(1+4*a^2))/2;
    temp = y_OLNM-oldgrad(y_OLNM)/L;
    y_OLNM = temp + (a-1)/na*(temp-z_OLNM);
    z_OLNM = temp;
    a = na;
    
    temp = y_ONM-grad(y_ONM)/L;
    y_ONM = temp+beta*(temp-x_ONM);
    x_ONM = temp;
end
track_OGD(m) = max(ite_OGD(track_idx));
track_ONM(m) = max(ite_ONM(track_idx));
track_OLNM(m) = max(ite_OLNM(track_idx));
end

figure
plot(sqrt(kappas),track_OGD,'ok-',sqrt(kappas),track_ONM,'^r-',sqrt(kappas),track_OLNM,'dg-','MarkerSize',10)
xlabel('Square root of condition number')
ylabel('Tracking iterate error')
lgd = legend('online gradient descent','online Nesterov`s method','OLNM');
lgd.Location = 'northwest';

%% Figure 3a
niter = 2e3;
d = 2;
mu = 1;
L = 5e2;
kappa = L/mu;
alpha = 2/(mu+L);
beta = (sqrt(kappa)-1)/(sqrt(kappa)+1);
sigma = 1;

A = [L 0; 0 mu];
f = @(x) .5*x'*A*x;

x0 = zeros(d,1);
x_OGD = x0;
x_ONM = x0;
y_ONM = x0;
x_OLNM = x0;
y_OLNM = x0;
z_OLNM = x0;
xopt = x0;
xopt(2) = sigma;

ite_OGD = zeros(niter,1);
ite_ONM = zeros(niter,1);
ite_OLNM = zeros(niter,1);

T = floor((2+sqrt(2))*sqrt(kappa));

for i = 1:niter
    disp(i)
    
    grad = @(x) A*(x-xopt);
    ft = @(x) f(x-xopt);
    
    x_OGD = x_OGD - alpha*grad(x_OGD);
    
    if mod(i-1,T) == 0
        a = 1;
        oldgrad = grad;
        x_OLNM = z_OLNM;
        y_OLNM = z_OLNM;
    end
    na = (1+sqrt(1+4*a^2))/2;
    temp = y_OLNM-oldgrad(y_OLNM)/L;
    y_OLNM = temp + (a-1)/na*(temp-z_OLNM);
    z_OLNM = temp;
    a = na;
    
    temp = y_ONM-grad(y_ONM)/L;
    y_ONM = temp+beta*(temp-x_ONM);
    x_ONM = temp;
    
    xopt(2) = xopt(2)+sigma;
    
    ite_OGD(i) = norm(x_OGD-xopt);
    ite_ONM(i) = norm(x_ONM-xopt);
    ite_OLNM(i) = norm(x_OLNM-xopt);
end

marker = linspace(niter/5,niter,5);
figure
plot(1:niter,ite_OGD,'ok-',1:niter,ite_ONM,'^r-',1:niter,ite_OLNM,'dg-','MarkerSize',10,'MarkerIndices',marker)
lgd = legend('online gradient descent','online Nesterov`s method','OLNM');
lgd.Location = 'northwest';
xlabel('Number of iterations')
ylabel('Iterate error')

%% Figure 3b
d = 2;
L = 1;
sigma = 1;
l = 10;
kappas = linspace(1,1e5,l);
track_OLNM = zeros(l,1);
for m = 1:l
kappa = kappas(m);
mu = L/kappa;
A = [L 0; 0 mu];
f = @(x) .5*x'*A*x;

x0 = zeros(d,1);
x_OLNM = x0;
y_OLNM = x0;
z_OLNM = x0;
xopt = x0;
xopt(2) = sigma;

T = floor((2+sqrt(2))*sqrt(kappa));
niter = 5*T;
track_idx = 4*T:niter;
ite_OLNM = zeros(niter,1);

for i = 1:niter
    disp([m i])
    
    grad = @(x) A*(x-xopt);
    ft = @(x) f(x-xopt);
    
    if mod(i-1,T) == 0
        a = 1;
        oldgrad = grad;
        x_OLNM = z_OLNM;
        y_OLNM = z_OLNM;
    end
    na = (1+sqrt(1+4*a^2))/2;
    temp = y_OLNM-oldgrad(y_OLNM)/L;
    y_OLNM = temp + (a-1)/na*(temp-z_OLNM);
    z_OLNM = temp;
    a = na;
    
    xopt(2) = xopt(2)+sigma;
    
    ite_OLNM(i) = norm(x_OLNM-xopt);
end
track_OLNM(m) = max(ite_OLNM(track_idx));
end

figure
plot(sqrt(kappas),track_OLNM,'dg-','MarkerSize',10)
xlabel('Square root of condition number')
ylabel('Tracking iterate error for OLNM')

%% Figure 4
cycles = 5;
L = 1;
d = 5;
n = 20;
trials = 200;
sigma = 1;
kappas = 2e3:2e3:1e4;
l = length(kappas);
track_OGD = zeros(l,trials);
track_ONM = track_OGD;
track_OLNM = track_OGD;
for m = 1:l
kappa = kappas(m);
mu = 1/kappa;
alpha = 2/(mu+L);
beta = (sqrt(kappa)-1)/(sqrt(kappa)+1);
T = floor((2+sqrt(2))*sqrt(kappa));
niter = cycles*T;
track_idx = (cycles-1)*T:niter;
for trial = 1:trials
[Q,R] = qr((randn(n)));
U = Q*diag(diag(R)./abs(diag(R)));
[Q,R] = qr((randn(d)));
V = Q*diag(diag(R)./abs(diag(R)));
D = linspace(1,1/sqrt(kappa),d);
S = [diag(D); zeros(n-d,d)];
A = U*S*V';

x0 = ones(d,1);
x_OGD = x0;
x_ONM = x0;
y_ONM = x0;
x_OLNM = x0;
y_OLNM = x0;
z_OLNM = x0;

ite_OGD = zeros(niter,1);
ite_ONM = zeros(niter,1);
ite_OLNM = zeros(niter,1);

xopt = ones(d,1);
noise = 1e-3;
b = A*xopt+noise*randn(n,1);

for i = 1:niter
    disp([m trial niter-i])
    
    grad = @(x) A'*(A*x-b);
    xopt = (A'*A)\(A'*b);
    f = @(x) .5*norm(A*x-b)^2;
    
    ite_OGD(i) = norm(x_OGD-xopt);
    ite_ONM(i) = norm(x_ONM-xopt);
    ite_OLNM(i) = norm(x_OLNM-xopt);

    j = randi(d,1);
    k = randi(2,1);
    xopt(j) = xopt(j) + sigma*(-1)^k;
    b = A*xopt+noise*randn(n,1);
    
    x_OGD = x_OGD - alpha*grad(x_OGD);
    
    if mod(i-1,T) == 0
        a = 1;
        oldgrad = grad;
%         x_OLNM = z_OLNM;
        y_OLNM = z_OLNM;
    end
    na = (1+sqrt(1+4*a^2))/2;
    temp = y_OLNM-oldgrad(y_OLNM)/L;
    y_OLNM = temp + (a-1)/na*(temp-z_OLNM);
    z_OLNM = temp;
    x_OLNM = z_OLNM; % modification
    a = na;
    
    temp = y_ONM-grad(y_ONM)/L;
    y_ONM = temp+beta*(temp-x_ONM);
    x_ONM = temp;
end
track_OGD(m,trial) = max(ite_OGD(track_idx));
track_ONM(m,trial) = max(ite_ONM(track_idx));
track_OLNM(m,trial) = max(ite_OLNM(track_idx));
end
end

figure
plot(kappas,mean(track_OGD,2),'ok',kappas,mean(track_ONM,2),'^r',kappas,mean(track_OLNM,2),'dg','MarkerSize',10)
xlabel('Condition number')
ylabel('Sample mean of tracking iterate error')
xlim([1900 10100])
ylim([5 30])
lgd = legend('online gradient descent','online Nesterov`s method','OLNM');
lgd.Location = 'northwest';

%% Figure 5
d = 5;
n = 20;
Ls = 2e2:2e2:1e3;
l = length(Ls);
phi = @(z) 1./(1+exp(-z));
xr = zeros(d,1);
niter_fixedpt = 5;

%% Figure 5a
try
    load('sigmaRdelta.mat','sigmaRdelta')
    sigmas = sigmaRdelta(:,1);
    Rs = sigmaRdelta(:,2);
    deltas = sigmaRdelta(:,3);
catch
sigmaRdelta = zeros(l,3);
trials = 50;
niter = 20;
eps = 1e-3; % stopping criteria for computing xopt
for m = 1:5
L = Ls(m);
for iter_fixedpt = 1:niter_fixedpt
if iter_fixedpt == 1
    delta = 0;
else
    delta = L*sqrt(sigma/2/R);
end
beta = (sqrt(L+delta)-sqrt(delta))/(sqrt(L+delta)+sqrt(delta));
sigmas = zeros(trials,1);
Rs = zeros(trials,1);
for trial = 1:trials
disp([m trial])
% compute A for the trial
[Q,R] = qr((randn(n)));
U = Q*diag(diag(R)./abs(diag(R)));
[Q,R] = qr((randn(d)));
V = Q*diag(diag(R)./abs(diag(R)));
D = rand(d,1);
D = D*sqrt(2*sqrt(L))/max(D);
S = [diag(D); zeros(n-d,d)];
A = U*S*V';

% for the trial, compute b at every time step
b = 2*randi(2,[n,1])-3;
bs = zeros(n,niter);
bs(:,1) = b;
for i = 2:niter
    % at each time step, flip a component of b
    bs(:,i) = bs(:,i-1);
    j = randi(n,1);
    bs(j,i) = -bs(j,i);
end
% for the trial, compute xopt at every time step
xopts = zeros(d,niter);
for i = 1:niter
    b = bs(:,i);
    psi = @(x) phi(b.*(A*x));
    grad = @(x) -A'*(b.*(1-psi(x)))+delta*(x-xr);
    xopt = zeros(d,1);
    yopt = xopt;
    iter = 0;
    % compute xopt using Nesterov's method
    while norm(grad(xopt))>=eps
        temp = yopt-grad(yopt)/L;
        yopt = temp+min([iter/(iter+3) beta])*(temp-xopt);
        xopt = temp;
        iter = iter+1;
    end
    xopts(:,i) = xopt;
end
sigma = 0;
% compute sigma
for i = 2:niter
    sigma = max([sigma norm(xopts(:,i)-xopts(:,i-1))]);
end
R = 0;
% compute R
for i = 1:niter
    R = max([R norm(xopts(:,i))]);
end
% compute delta
sigmas(trial) = sigma;
Rs(trial) = R;
end
sigma = mean(sigmas);
R = mean(Rs);
end
sigmaRdelta(m,:) = [sigma R delta];
end
sigmas = sigmaRdelta(:,1);
Rs = sigmaRdelta(:,2);
deltas = sigmaRdelta(:,3);
save('sigmaRdelta.mat','sigmaRdelta')
end

figure
plot(Ls,sigmas,'ok',Ls,Rs,'^r','MarkerSize',10)
legend('\sigma(\delta)','R(\delta)')
xlabel('L')
xlim([100 1100])
ylabel('Sample mean')

%% Figure 5b
recenter = 1; %1 means no recentering
trials = 200;
eps = 1e-3; % stopping criteria for tracking
win = 100; % width of tracking window

track_OGD = zeros(l,trials);
track_ORGD = zeros(l,trials);
track_triv = zeros(l,trials);

for m = 1:l
L = Ls(m);
delta = deltas(m);
for trial = 1:trials
disp([m trial])
[Q,R] = qr((randn(n)));
U = Q*diag(diag(R)./abs(diag(R)));
[Q,R] = qr((randn(d)));
V = Q*diag(diag(R)./abs(diag(R)));
D = rand(d,1);
D = D*sqrt(2*sqrt(L))/max(D);
S = [diag(D); zeros(n-d,d)];
A = U*S*V';

x0 = zeros(d,1);
x_OGD = x0;
x_ORGD = x0;

gra_triv = zeros(win,1);
gra_OGD = zeros(win,1);
gra_ORGD = zeros(win,1);

b = 2*randi(2,[n,1])-3;

crit = eps+1;
i = 1;
while crit >= eps
    disp([m trial])
    
    psi = @(x) phi(b.*(A*x));
    grad = @(x) -A'*(b.*(1-psi(x)));
    
    grad_OGD = grad(x_OGD);
    gra_OGD(i) = norm(grad_OGD);
    x_OGD = x_OGD-grad_OGD/L;
    
    if mod(i,recenter)==1
        xr = x_ORGD;
    end

    grad_ORGD = grad(x_ORGD);
    gra_ORGD(i) = norm(grad_ORGD);
    x_ORGD = x_ORGD-(grad_ORGD+delta*(x_ORGD-xr))/(L+2*delta)*2;
    
    gra_triv(i) = norm(grad(x0));
    
    j = randi(n,1);
    b(j) = -b(j);
    
    if mod(i,win)==0 && i>=2*win
        quo = i/win;
        i1 = (quo-2)*win+1;
        i2 = (quo-1)*win+1;
        crit_OGD = (max(gra_OGD(i1:i))-max(gra_OGD(i2:i)))/gra_OGD(1);
        crit_ORGD = (max(gra_ORGD(i1:i))-max(gra_ORGD(i2:i)))/gra_ORGD(1);
        crit = max(crit_OGD,crit_ORGD);
    end
    i = i+1;
end
track_OGD(m,trial) = max(gra_OGD(i-win:i-1));
track_ORGD(m,trial) = max(gra_ORGD(i-win:i-1));
track_triv(m,trial) = max(gra_triv);
end
end

figure
plot(Ls,mean(track_OGD,2),'ok',Ls,mean(track_ORGD,2),'dg',Ls,mean(track_triv,2),'^r','MarkerSize',10)
xlabel('L')
ylabel('Sample mean of tracking gradient error')
xlim([100 1100])
lgd = legend('online gradient descent','ORGD','x_c');
lgd.Location = 'northwest';
