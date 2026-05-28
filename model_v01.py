import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy import arccos, cos, pi, sin


def init_image():
    fig, ax = plt.subplots()
    ax.set_xlim(-16, 16)
    ax.set_ylim(-5, 16)
    ax.set_aspect("equal")
    (line1,) = ax.plot([], [], color="r", linewidth=4)
    (line2,) = ax.plot([], [], color="g", linewidth=4)
    return fig, ax, line1, line2


def joint_angles_to_xy(th1, th2):
    L1, L2 = 10, 5
    x1 = L1 * cos(th1)
    y1 = L1 * sin(th1)
    x2 = x1 + L2 * cos(th1 + th2)
    y2 = y1 + L2 * sin(th1 + th2)
    return np.column_stack([x1, y1]), np.column_stack([x2, y2])


def update_image(frame):
    shoulder = (0, 0)
    elbow, wrist = frame
    x0, y0 = shoulder
    x1, y1 = elbow
    x2, y2 = wrist
    line1.set_data([x0, x1], [y0, y1])
    line2.set_data([x1, x2], [y1, y2])
    return line1, line2


def target_xy_to_joint_angles(x, y):
    L1, L2 = 12, 4
    th2 = arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))
    th1 = arccos(((L1 + L2 * cos(th2)) * x + L2 * sin(th2) * y) / (x**2 + y**2))
    return th1, th2


# create target x-y pairs
n_pts = 1601
# x_targ = np.linspace(-8,8,n_pts)
# y_targ = 12 + 0*x_targ
# for some reason the math doesn't work if we start at 0 degrees
# need to start closer to 20 for some reason
# oh, I think its a consequence of th1 needing to be negative
# ... deal with that later
x_targ = 13 * cos(np.linspace(20 * pi / 180, pi, num=n_pts))
y_targ = 13 * sin(np.linspace(20 * pi / 180, pi, num=n_pts))

# ideally, target xy values will be tracked perfectly by the arm
# but this only happens if the xy->angles function uses same L1,L2
# as angles->xy. Solution is to put a cerebellum in the middle.

th1_values, th2_values = target_xy_to_joint_angles(x_targ, y_targ)

elbow, wrist = joint_angles_to_xy(th1_values, th2_values)

# create animation
fig, ax, line1, line2 = init_image()
ax.plot(x_targ, y_targ, "k--")
step = 25
anim = FuncAnimation(
    fig,
    update_image,
    frames=zip(elbow[::step], wrist[::step]),
    save_count=n_pts,
    interval=50,
    blit=True,
)

# on vscode install "Live Server", then
# right-click and "Open with Live Server"
anim.save("animation.html", writer="html")
plt.close()
