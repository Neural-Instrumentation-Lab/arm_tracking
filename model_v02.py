"""
Cerebellar motor learning simulation of a 2-joint planar arm.

Identical to model_v01.py except the Cerebellum class is replaced by a C++
implementation compiled via pybind11.  Build the extension first:

    python setup.py build_ext --inplace

Then run:

    python model_v02.py        # writes animation.html
    # open animation.html with Live Server in VS Code
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy import arccos, cos, pi, sin, arctan2
from types import SimpleNamespace
from cerebellum import Cerebellum  # C++ extension via pybind11


def trajectory_001():
    """
    Semicircular arc from ~20° to 180° at radius 13.

    Returns
    -------
    x_target : ndarray of shape (n_pts, 2)
        Target positions [x1, x2].
    n_pts : int
        Number of trajectory points.
    """
    n_pts = 1601
    x1 = 13 * cos(np.linspace(20 * pi / 180, pi, num=n_pts))
    x2 = 13 * sin(np.linspace(20 * pi / 180, pi, num=n_pts))
    return np.column_stack([x1, x2]), n_pts


def trajectory_002():
    """
    Horizontal back-and-forth sweep between (-8, 12) and (8, 12), 5 cycles.

    Returns
    -------
    x_target : ndarray of shape (n_pts, 2)
        Target positions [x1, x2].
    n_pts : int
        Number of trajectory points.
    """
    n_pts = 1600
    n_cycles = 5
    ppc = n_pts // n_cycles
    x1_cycle = np.concatenate(
        [np.linspace(-8, 8, ppc // 2), np.linspace(8, -8, ppc // 2)]
    )
    x1 = np.tile(x1_cycle, n_cycles)[:n_pts]
    x2 = np.full(n_pts, 12)
    return np.column_stack([x1, x2]), n_pts


def trajectory_003():
    """
    L-shaped path: horizontal (-8,12)→(8,12), then vertical (8,12)→(8,0).
    Points distributed proportionally to segment length.

    Returns
    -------
    x_target : ndarray of shape (n_pts, 2)
        Target positions [x1, x2].
    n_pts : int
        Number of trajectory points.
    """
    n_pts = 1601
    # split points proportionally to segment length: 16 units horiz, 12 units vert
    n1 = int(n_pts * 16 / 28)
    n2 = n_pts - n1
    x1 = np.concatenate([np.linspace(-8, 8, n1), np.full(n2, 8)])
    x2 = np.concatenate([np.full(n1, 12), np.linspace(12, 0, n2)])
    return np.column_stack([x1, x2]), n_pts


def trajectory_004():
    """
    Repeated over-down-up-back cycles: (-8,12)→(8,12)→(8,0)→(8,12)→(-8,12), 5 times.
    Points distributed proportionally to segment length within each cycle.

    Returns
    -------
    x_target : ndarray of shape (n_pts, 2)
        Target positions [x1, x2].
    n_pts : int
        Number of trajectory points.
    """
    n_pts = 1600
    # one cycle: over [16], down [12], up [12], back [16] = 56 units total
    n_cycles = 5
    ppc = n_pts // n_cycles  # points per cycle
    n1 = round(ppc * 16 / 56)
    n2 = round(ppc * 12 / 56)
    n3 = round(ppc * 12 / 56)
    n4 = ppc - n1 - n2 - n3
    x1_cycle = np.concatenate(
        [
            np.linspace(-8, 8, n1),  # over
            np.full(n2, 8),  # down
            np.full(n3, 8),  # up
            np.linspace(8, -8, n4),  # back
        ]
    )
    x2_cycle = np.concatenate(
        [
            np.full(n1, 12),  # over
            np.linspace(12, 0, n2),  # down
            np.linspace(0, 12, n3),  # up
            np.full(n4, 12),  # back
        ]
    )
    x1 = np.tile(x1_cycle, n_cycles)[:n_pts]
    x2 = np.tile(x2_cycle, n_cycles)[:n_pts]
    return np.column_stack([x1, x2]), n_pts


def init_image(n_pts, max_err):
    """
    Initialize the matplotlib figure, axes, and animated artists.

    Parameters
    ----------
    n_pts : int
        Total number of trajectory points (sets error plot x-axis).
    max_err : float
        Maximum error value (sets error plot y-axis).

    Returns
    -------
    SimpleNamespace with fields: fig, ax_arm, ax_err, line1, line2,
    target_marker, err_cursor.
    """
    fig, (ax_arm, ax_err) = plt.subplots(1, 2, figsize=(12, 5))
    ax_arm.set_xlim(-16, 16)
    ax_arm.set_ylim(-5, 16)
    ax_arm.set_aspect("equal")
    (line1,) = ax_arm.plot([], [], color="r", linewidth=4)
    (line2,) = ax_arm.plot([], [], color="g", linewidth=4)
    (target_marker,) = ax_arm.plot(
        [], [], "x", color="b", markersize=10, markeredgewidth=2
    )
    ax_err.set_xlim(0, n_pts)
    ax_err.set_ylim(0, max_err * 1.1)
    ax_err.set_xlabel("time step")
    ax_err.set_ylabel("error (L2)")
    (err_cursor,) = ax_err.plot([0, 0], [0, max_err * 1.1], color="r", linewidth=1)
    return SimpleNamespace(
        fig=fig,
        ax_arm=ax_arm,
        ax_err=ax_err,
        line1=line1,
        line2=line2,
        target_marker=target_marker,
        err_cursor=err_cursor,
    )


def P(m):
    """
    Forward model ("plant") converting motor angles to Cartesian positions.

    Parameters
    ----------
    m : array-like of length 2
        Motor angles [m1, m2] in radians.

    Returns
    -------
    elbow : ndarray of shape (N, 2)
        Elbow (x, y) positions.
    wrist : ndarray of shape (N, 2)
        Wrist (x, y) positions.
    """
    L1, L2 = 10, 5
    m1, m2 = m[0], m[1]
    elbow_x1 = L1 * cos(m1)
    elbow_x2 = L1 * sin(m1)
    wrist_x1 = elbow_x1 + L2 * cos(m1 + m2)
    wrist_x2 = elbow_x2 + L2 * sin(m1 + m2)
    return np.column_stack([elbow_x1, elbow_x2]), np.column_stack([wrist_x1, wrist_x2])


def B(xd, c):
    """
    Inverse model ("brainstem") converting Cartesian positions to motor angles.

    Parameters
    ----------
    xd : array-like of length 2
        Desired target position [x1, x2].
    c : array-like of length 2
        Cerebellar correction [c1, c2].

    Returns
    -------
    m : ndarray of length 2
        Motor angles [m1, m2] in radians.
    """
    # L1, L2 = 10, 5
    L1, L2 = 10.5, 4.1
    # L1, L2 = 9, 6
    x1 = xd[0] + c[0]
    x2 = xd[1] + c[1]
    m2 = arccos((x1**2 + x2**2 - L1**2 - L2**2) / (2 * L1 * L2))
    m1 = arctan2(x2, x1) - arctan2(L2 * sin(m2), L1 + L2 * cos(m2))
    return np.array([m1, m2])


def computeErr(x_desired, x_actual):
    """
    Compute endpoint error between desired and actual wrist position.

    Parameters
    ----------
    x_desired : array-like of length 2
        Desired [x1, x2] position.
    x_actual : array-like of length 2
        Actual [x1, x2] position.

    Returns
    -------
    err : ndarray of length 2
        Signed error (actual - desired).
    """
    err = x_actual - x_desired
    return err


def main():
    """
    Run the cerebellar learning simulation and save the animation to animation.html.

    Selects a trajectory, simulates the arm with online cerebellar adaptation,
    then renders a two-panel animation: the arm tracking the target on the left,
    and endpoint error (with and without cerebellum) on the right.
    """
    x_target, n_pts = trajectory_001()
    m_values = np.zeros((n_pts, 2))
    el_values = np.zeros((n_pts, 2))
    wr_values = np.zeros((n_pts, 2))
    err_values = np.full(n_pts, np.nan)
    err_no_cb = np.array(
        [np.linalg.norm(computeErr(x, P(B(x, np.zeros(2)))[1])) for x in x_target]
    )

    C = Cerebellum()
    for i in range(1, n_pts):
        xd = x_target[i]
        m = m_values[i - 1, :]
        c = C.compute(m)
        m = B(xd, c)
        elbow, wrist = P(m)
        err = computeErr(xd, wrist)
        C.update(err)

        m_values[i] = m
        el_values[i] = elbow
        wr_values[i] = wrist
        err_values[i] = np.linalg.norm(err)

    a = init_image(n_pts, max(np.nanmax(err_values), np.nanmax(err_no_cb)))
    a.ax_arm.plot(x_target[:, 0], x_target[:, 1], "k--")
    a.ax_err.plot(
        range(n_pts),
        err_no_cb,
        color="gray",
        linewidth=1,
        linestyle="--",
        label="no cerebellum",
    )
    a.ax_err.plot(
        range(n_pts), err_values, color="k", linewidth=1, label="with cerebellum"
    )
    a.ax_err.legend(loc="upper right")

    def update_image(frame):
        elbow, wrist, target, idx = frame
        a.line1.set_data([0, elbow[0]], [0, elbow[1]])
        a.line2.set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
        a.target_marker.set_data([target[0]], [target[1]])
        a.err_cursor.set_xdata([idx, idx])
        return a.line1, a.line2, a.target_marker, a.err_cursor

    step = 25
    indices = range(0, n_pts, step)
    targets = np.column_stack([x_target[:, 0], x_target[:, 1]])
    anim = FuncAnimation(
        a.fig,
        update_image,
        frames=zip(el_values[::step], wr_values[::step], targets[::step], indices),
        save_count=len(indices),
        interval=100,
        blit=True,
    )

    # open animation.html with Live Server in VS Code (right-click → Open with Live Server)
    anim.save("animation.html", writer="html")
    plt.close()


if __name__ == "__main__":
    main()
