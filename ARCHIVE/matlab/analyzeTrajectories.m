% green data
xdata = xdata - mean(xdata);
ydata = ydata - mean(ydata);

figure(1); clf;
tt = (1:length(xdata))/126.5;
plot(tt,xdata,tt,ydata);

figure(2); clf;
plot(xdata(tt>50),ydata(tt>50),'k'); grid on;
axis([-4 4 -100 100]);


%%

fs = 40e3;
dt = 1/fs;
w = 3*pi/32;
T = (2*pi)/w;
t = 0:dt:T;
m = 4;

a = xx(2,1) - mean(xx(:,1));
b = yy(1,2) - mean(yy(:,2));

ww = -a / sqrt(m^2 - a^2);
wc = w / ww;
fc = wc / (2*pi);
k = b*(ww^2 + 1) / (m*ww);

x = m*cos(w*t);
M = 1/sqrt(ww^2 + 1);
P = -atan(ww);
y = k*m*M*cos(w*t + P);

am = mean(xx(:,1)) ; bm = mean(yy(:,2));
figure(2); hold on;
h = plot(x,y,'g');

%%

delete(findobj(gca,'color','r'));
yhat = k*m*M*sin(w*tt+P);
figure(1); hold on
plot(tt,abs(yhat).*erf(yhat/400),'r');
