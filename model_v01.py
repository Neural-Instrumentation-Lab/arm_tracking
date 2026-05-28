# %%
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    _ipy = get_ipython()
    _in_notebook = _ipy is not None
except NameError:
    _in_notebook = False

if _in_notebook:
    _ipy.run_line_magic('matplotlib', 'inline')
    from IPython.display import HTML


th2_values = np.linspace(0, pi, 60)
# th1_values = np.linspace(0, pi/2, 60)
th1_values = np.zeros_like(th2_values) + pi/4

def init_image():
    fig, ax = plt.subplots()
    ax.set_xlim(-16, 16)
    ax.set_ylim(-5, 16)
    line1, = ax.plot([], [], color='r', linewidth=4)
    line2, = ax.plot([], [], color='g', linewidth=4)
    return fig, line1, line2

def joint_angle_to_xy(th1,th2):
    L1, L2 = 10, 5
    x1 = L1*cos(th1)
    y1 = L1*sin(th1)
    x2 = x1 + L2*cos(th1+th2)
    y2 = y1 + L2*sin(th1+th2)
    return (x1,y1) , (x2,y2)

shoulder = (0,0)
def update_image(frame):
    th1, th2 = frame
    elbow , wrist = joint_angle_to_xy(th1,th2)
    x0,y0 = shoulder
    x1,y1 = elbow
    x2,y2 = wrist
    line1.set_data([x0, x1], [y0, y1])
    line2.set_data([x1, x2], [y1, y2])
    return line1, line2

# run model and create animation
fig, line1, line2 = init_image()
anim = FuncAnimation(fig, update_image,
                     frames=zip(th1_values, th2_values),
                     save_count=len(th2_values),
                     interval=50,
                     blit=True)

if _in_notebook:
    plt.close()
    HTML(anim.to_jshtml())
else:
    plt.show()

# %%
