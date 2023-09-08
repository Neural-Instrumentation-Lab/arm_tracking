clear;clf;

global FS;
FS = 1e3;
dt = 1/FS;
tMax = 10;
nPts = tMax*FS;
t = (1:nPts)/FS;
nCtrs = 128;

f = 1;
vDesired  = cos(2*pi*f*t);
vCommand  = 0*vDesired;
vRealized = 0*vDesired;

ctrmx = 3;
c.ctr = linspace(-ctrmx,ctrmx,nCtrs);
c.wts = 0*ones(1,nCtrs);
c.sig = ctrmx/(nCtrs-1);
c.beta = 0.05;
c.p    = zeros(1,nCtrs);

nCerebellums = 50;
for i = 1:nCerebellums
    cx(i) = c;
end

for i = nCerebellums:nPts
    corr = 0;
    for j = 1:nCerebellums
        [cx(j),corrTmp] = stim(vDesired(i-j+1) , cx(j));
        corr = corr + corrTmp;
    end
    
    vCommand(i) = vDesired(i)  + corr;
    vRealized(i) = plant(vCommand(i));
    
    err = vRealized(i) - vDesired(i);
    
    for j = 1:nCerebellums
        cx(j) = updateErrs(err,cx(j));
    end
    
end

plot(t,vDesired,t,vRealized);
% plot(t,vDesired-vRealized);

axis([0 t(end) -2 2]);
