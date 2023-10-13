clear;clf;
%%
hh = get(gca,'children');
h = hh(end);
x = get(h,'XData');
y = get(h,'YData');

%%
figure(3);clf;
plot(y,x);