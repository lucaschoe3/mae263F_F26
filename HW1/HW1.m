clc;
clear all;
close all;
%%MAE 263F HW 1
%%Lucas Choe


%% defining parameters
%change these values to change system parameters

N= 1; %%DOF
% masses
m1 = 1; m2 = 2; m3=3 ;m4 = 1; m5 = 1;
%spring constants
k1= 1; k2 = 20; k3 = 30; k4 = 10; k5 = 10;
%damping values
b1 = 0; b2 = 10; b3 = 15; b4 = 5; b5 = 5;
dt = 0.1;
free = 2:N+1;
T = 100;
tol = 0.01;

%% splitting into Params
param.N = N;
param.m = [0, m1]';
param.k = [0, k1]';
param.b = [0, b1]';
param.g = 9.8;
param.dt = dt;
param.eps = tol;
param.free = free;


%% initial pos
x_init = zeros(N+1,1);
u_init = zeros(N+1,1);
x = zeros(N+1,T/dt);
x(:,1) = x_init;
u= zeros(N+1, T/dt);
u(:,1) = u_init;




%% required Functions 
%Gradient
function Grad = gradEs(xa,xb,k)
    Grad = k*(xb-xa)*[-1 ; 1];
end

%Hessian
function Hess = hessEs(k)
    Hess = k*[[1,-1];[-1,1]];
end

%Damping
function Damp = dampingForce(xa_new,xb_new, xa_old, xb_old, b, dt)
    vela = (xa_new-xa_old)/dt;
    velb = (xb_new-xb_old)/dt;
    Damp = b*(velb-vela)*[-1;1];
end

%damping Jacobian
function DampJ = dampingJacobian(b,dt)
    DampJ = (b/dt)*[[1, -1]; [-1, 1]];
end

%residual F and Jacobian
function [f,J] = getFJ(x_new, x_old, u_old, param)
 
    %extracting params
    m= param.m; k=param.k; b=param.b; g=param.g; dt=param.dt;N=param.N;
    

    %initializing f and J
    %f= diag((m./dt).*u_new) + diag(-m.*g);
    f = (m./dt).*((x_new-x_old)./dt-u_old)-m.*g;
    J = diag(m)./dt^2;
    for i = 1:N
        idx = [i, i+1];
        %adding spring forces
        f(idx) = f(idx) + gradEs(x_new(i), x_new(i+1), k(i+1));
        J(idx,idx) = J(idx,idx) + hessEs(k(i+1));
        %adding damper forces
        f(idx)= f(idx) + dampingForce(x_new(i), x_new(i+1), x_old(i), x_old(i+1), b(i+1), dt);
        J(idx,idx) = J(idx,idx) + dampingJacobian(b(i+1),dt);
    end
end

%implicit Euler Step
function [x_new, u_new] = implicitEulerStep(x_old,u_old,param)
    x_new = x_old;
    eps = param.eps;
    err=inf;
    free = param.free;
    while err>eps
        [f,J]=getFJ(x_new, x_old, u_old, param);
        dx = (J(free,free)\f(free))';
        x_new(free) = x_new(free) - dx';
        [f,~] = getFJ(x_new,x_old,u_old,param);
        err = abs(f(free));

    end
    u_new = (x_new-x_old)/param.dt;
end


%% Explicit Euler
function [x_new,u_new] = explicitEulerStep(x_old, u_old, param)
dt = param.dt;
m = param.m;
g = param.g;
b= param.b;
N = param.N;
k = param.k;
F = zeros(1,N+1);
    for i = 1:N
        idx = [i; i+1];
        %adding spring forces
        F(idx) = F(idx) + gradEs(x_old(i), x_old(i+1), k(i+1));

        %adding damper forces
        F(idx)= F(idx) + b(i).*(u_old(i+1)-u_old(i));
    end
    free = param.free;
    a = zeros(1,N+1);
    a(free) = g - F(free)./m(free);
    x_new = x_old + dt.*u_old;
    u_new = (u_old + dt.*a);
end


%% Main simulation

for k = 1:T/dt-1
    %%[x(:,k+1), u(:,k+1)] = implicitEulerStep(x(:,k), u(:,k), param);
    [x(:,k+1), u(:,k+1)] = explicitEulerStep(x(:,k), u(:,k), param);

end

%% PLot
t = 0:dt:T;


figure (1); 

plot(t,x(2,:), 'LineWidth', 1.25, 'Color', 'orange');
xlabel('Time (s)');
ylabel('Position (m)')
title('Positions of Each Mass')
grid on;
hold on
figure(2); 
plot(t,x(2:end,:), 'LineWidth', 1.25);
xlabel('Time (s)');
ylabel('Velocity (m/s)')
title('Velocities of Each Mass')
grid on;
