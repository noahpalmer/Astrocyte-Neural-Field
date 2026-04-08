% Finite difference simulation of astrocyte neural field model.
%
% Runs a simulation of the astrocyte-neural field model for specified initial conditions 
% Currently setup to run a simulation with IC near the stationary bump solution
% derived for the model with a right shift perturbation.


% Spatial and temporal gride and spacing.
N = 3000;    
dx = 2*pi/N;    
x = linspace(-pi,pi-dx,N)'; 

dt = 0.01;  
T = 200;    
nt = round(T/dt)+1;


% Model parameters
gamma = 2; % Synaptic replenishment rate
beta = 0.05; % Synaptic depletion rate
tau = 1; % Synaptic depression timescale
D = 0.7; % Astrocytic resource diffusion constant
theta = 0.1; %Neural activity threshold


% Use threshold condition to find the bump half-width

f = @(delta) czero(delta,beta,gamma).*sin(2*delta)-theta;
initial_guess = 1.5; 
Delta = fzero(f, initial_guess);

% Compute astrocyte resource amplitude and synaptic resource amplitude in the active region for
% stationary bump solutions.

kappa = Delta/pi;
c0 = czero(Delta,beta,gamma);
A0 = kappa*(1-c0);
epsilon = 0.3*(2*c0*sin(Delta));


% Initial conditions: stationary bump profile of section 3

U = zeros(N,nt); Q = ones(N,nt); A = U;
U(:,1) = (2*c0*sin(Delta)).*cos(x)+epsilon.*sin(x);
A(:,1) = A0;
Q(:,1) = (c0).*((x>-Delta) & (x<Delta))+1.*((x>Delta)|(x<-Delta));

% Diffusion operator with periodic BCs

e = ones(N,1);
D2 = spdiags([e -2*e e], -1:1, N, N);
D2(1,end) = 1;  
D2(end,1) = 1;  
D2 = D2/dx^2;
IA = speye(N);
LA = IA - (dt*D/tau)*D2;

% Precompute sine and cosine terms times dx needed for integral term in the equation for u(x,t).

cx = dx*cos(x'); sx = dx*sin(x');


for k=1:nt -1

    %u(x,t) update
    QHu = Q(:,k).*(U(:,k) > theta);
    fcos = cx*QHu; fsin = sx*QHu;
    U(:,k+1) = (1-dt)*U(:,k)+dt*(fcos*cos(x)+fsin*sin(x));
    
    % q(x,t) update, first compute integrating factor for q(x,t) 
    c = gamma*A(:,k);
    lam = c+beta*(U(:,k) > theta);
    E = exp(-(dt/tau)*lam);

    Q(:,k+1) = E.*Q(:,k) + (1 - E).*(c./max(lam,1e-12));

    % a(x,t) update, backward euler
    Rexp = (dt/tau)*(beta*QHu-gamma*A(:,k).*(1-Q(:,k)));
    A(:,k+1) = LA\(A(:,k)+Rexp);
end


function c0 = czero(delta, beta, gamma)  
    c0 = (beta+2*(gamma*delta/pi)-sqrt(beta^2+4*beta*gamma*delta/pi))./(2*(gamma*delta/pi));
end


%%


% Code in this section creates Figures 2 and 3.



% Creates the plots for Figure 7. Requires first running a finite difference simulation with a perturbation to the activity variable.

figure;
t = tiledlayout(2,3);

ax4 = nexttile;
plot(ax4, x, Q(:,70), '-','Color',[0 0.6 0.3 1], 'LineWidth',7.5); hold on;
plot(ax4,x, Q(:,1), ':','Color',[0 0.6 0.3 0.5],'LineWidth',7.5)
xlim([-pi,pi])
ylim([0.77,1.05])

ax5 = nexttile;
plot(ax5, x, Q(:,3000), '-','Color',[0 0.6 0.3 1], 'LineWidth',7.5); hold on;
plot(ax5,x, Q(:,1), ':','Color',[0 0.6 0.3 0.5],'LineWidth',7.5)
xlim([-pi,pi])
ylim([0.77,1.05])


ax6 = nexttile;
plot(ax6, x, Q(:,10000), '-','Color',[0 0.6 0.3 1], 'LineWidth',7.5); hold on;
plot(ax6,x, Q(:,1), ':','Color',[0 0.6 0.3 0.5],'LineWidth',7.5)
xlim([-pi,pi])
ylim([0.77,1.05])


ax7 = nexttile;
plot(ax7, x, A(:,70), '-','Color',[0.85 0.35 0.75 1], 'LineWidth',7.5); hold on;
plot(ax7,x, A(:,1), ':','Color',[0.85 0.35 0.75 0.5],'LineWidth',7.5)
xlim([-pi,pi])
ylim([0.092,0.108])


ax8 = nexttile;
plot(ax8, x, A(:,3000), '-','Color',[0.85 0.35 0.75 1], 'LineWidth',7.5); hold on;
plot(ax8,x, A(:,1), ':','Color',[0.85 0.35 0.75 0.5],'LineWidth',7.5)
xlim([-pi,pi])
ylim([0.092,0.108])


ax9 = nexttile;
plot(ax9, x, A(:,10000), '-','Color',[0.85 0.35 0.75 1], 'LineWidth',7.5); hold on;
plot(ax9,x, A(:,1), ':','Color',[0.85 0.35 0.75 0.5],'LineWidth',7.5)
xlim([-pi,pi])
ylim([0.092,0.108])

