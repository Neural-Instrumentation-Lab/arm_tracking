function [c,val] = stim(vDesired,c)

c.p = exp((-(vDesired-c.ctr).^2) / (2*(c.sig)^2));
val = sum(c.wts .* c.p);



