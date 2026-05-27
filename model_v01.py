# %%
import numpy as np
from numpy import sin,cos,pi
import matplotlib.pyplot as plt
%matplotlib inline

Fs = 100
tmax = 16
nPts = Fs*tmax
t = np.arange(nPts)/Fs
xTarg = np.linspace(-8,8,nPts)
yTarg = 12 * np.ones(nPts)

L1 , L2 = 10, 5
th1 , th2 = 0, 0

x1 = L1*cos(th1)
y1 = L1*sin(th1)
x2 = x1 + L2*cos(th2)*cos(th1)
y2 = y1 + L2*cos(th2)*sin(th1)

# %%
plt.plot([0,x1] ,[0,y1]  , color='r' , linewidth=4)
plt.plot([x1,x2],[y1,y2] , color='g' , linewidth=4)
plt.xlim((-16,16))
plt.ylim((-5,16))
plt.show()

# %%
