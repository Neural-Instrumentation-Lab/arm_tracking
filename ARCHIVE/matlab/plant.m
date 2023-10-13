function val = plant(x)
persistent y
global FS

% initialize
if (isempty(y))
    y = 0;
end

fc = 1;
tau = 1/(2*pi*fc);
dt = 1/FS;
dy = (x-y)/tau;
y = y + dt*dy;

a = 0.5;
val = erf(y/a);
% val = y;
