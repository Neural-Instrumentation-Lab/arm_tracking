clear;clf;
global FS
FS = 2e3;
f = .5;
k = 1;
dt = 1/FS;
t = 0:dt:10;
x = cos(2*pi*f*t);
z = 0*x;
for i = 1:length(t)
    z(i) = plant(x(i));
end

plot(x,k*z);
axis(2*[-1.25 1.25 -1.25 1.25]); axis equal; grid on;

% plot(t,x,t,z);
