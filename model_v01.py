import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy import arccos, cos, pi, sin, arctan2


def trajectory_001():
    x1_target = 13 * cos(np.linspace(20 * pi / 180, pi, num=n_pts))
    x2_target = 13 * sin(np.linspace(20 * pi / 180, pi, num=n_pts))
    return x1_target, x2_target


def trajectory_002():
    x1_target = np.linspace(-8, 8, n_pts)
    x2_target = 12 + 0 * x1_target
    return x1_target, x2_target


def trajectory_003():
    # split points proportionally to segment length: 16 units horiz, 12 units vert
    n1 = int(n_pts * 16 / 28)
    n2 = n_pts - n1
    x1_target = np.concatenate([np.linspace(-8, 8, n1), np.full(n2, 8)])
    x2_target = np.concatenate([np.full(n1, 12), np.linspace(12, 0, n2)])
    return x1_target, x2_target


def init_image():
    fig, ax = plt.subplots()
    ax.set_xlim(-16, 16)
    ax.set_ylim(-5, 16)
    ax.set_aspect("equal")
    (line1,) = ax.plot([], [], color="r", linewidth=4)
    (line2,) = ax.plot([], [], color="g", linewidth=4)
    (target_marker,) = ax.plot([], [], "x", color="b", markersize=10, markeredgewidth=2)
    return fig, ax, line1, line2, target_marker


def P(m1, m2):
    """
    Forward model ("plant") converting motor angles to Cartesian positions.

    Parameters
    ----------
    m1 : float or array-like
        Shoulder joint angle in radians.
    m2 : float or array-like
        Elbow joint angle in radians.

    Returns
    -------
    elbow : ndarray of shape (N, 2)
        Elbow (x, y) positions.
    wrist : ndarray of shape (N, 2)
        Wrist (x, y) positions.
    """
    L1, L2 = 10, 5
    elbow_x1 = L1 * cos(m1)
    elbow_x2 = L1 * sin(m1)
    wrist_x1 = elbow_x1 + L2 * cos(m1 + m2)
    wrist_x2 = elbow_x2 + L2 * sin(m1 + m2)
    return np.column_stack([elbow_x1, elbow_x2]), np.column_stack([wrist_x1, wrist_x2])


def update_image(frame):
    shoulder = (0, 0)
    elbow, wrist, target = frame
    x0, y0 = shoulder
    x1, y1 = elbow
    x2, y2 = wrist
    line1.set_data([x0, x1], [y0, y1])
    line2.set_data([x1, x2], [y1, y2])
    target_marker.set_data([target[0]], [target[1]])
    return line1, line2, target_marker


def B(x1, x2):
    """
    Inverse model ("brainstem") converting Cartesian positions to motor angles.

    Parameters
    ----------
    x1 : float or array-like
        Target x position.
    x2 : float or array-like
        Target y position.

    Returns
    -------
    m1 : float or ndarray
        Shoulder joint angle in radians.
    m2 : float or ndarray
        Elbow joint angle in radians.
    """
    # L1, L2 = 10, 5
    L1, L2 = 10.1, 4.5

    m2 = arccos((x1**2 + x2**2 - L1**2 - L2**2) / (2 * L1 * L2))
    m1 = arctan2(x2, x1) - arctan2(L2 * sin(m2), L1 + L2 * cos(m2))
    return m1, m2


# main loop
n_pts = 1601
x1_target, x2_target = trajectory_003()
m1_values, m2_values = B(x1_target, x2_target)
elbow, wrist = P(m1_values, m2_values)


# create animation
fig, ax, line1, line2, target_marker = init_image()
ax.plot(x1_target, x2_target, "k--")
step = 25
targets = np.column_stack([x1_target, x2_target])
anim = FuncAnimation(
    fig,
    update_image,
    frames=zip(elbow[::step], wrist[::step], targets[::step]),
    save_count=n_pts,
    interval=50,
    blit=True,
)

# on vscode install "Live Server", then
# right-click and "Open with Live Server"
anim.save("animation.html", writer="html")
plt.close()
