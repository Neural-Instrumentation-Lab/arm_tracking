import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class JoystickDotSimulator:
    def __init__(self, max_speed=5.0, initial_pos=(0.0, 0.0)):
        """
        :param max_speed: Maximum units per second the dot can move.
        :param initial_pos: Starting (x, y) coordinates.
        """
        self.max_speed = max_speed
        self.pos = np.array(initial_pos, dtype=float)

    def getPos(self):
        return self.pos

    def step(self, joy_x: float, joy_y: float, dt: float) -> np.ndarray:
        """
        Simulates the movement of the dot for a single timestep.
        
        :param joy_x: Joystick X input in range [-1.0, 1.0]
        :param joy_y: Joystick Y input in range [-1.0, 1.0]
        :param dt: Timestep in seconds
        :return: Updated (x, y) position
        """
        joy_vector = np.array([joy_x, joy_y], dtype=float)
        magnitude = np.linalg.norm(joy_vector)
        
        # Clamp joystick magnitude to max 1.0
        # if magnitude > 1.0:
        #     joy_vector = joy_vector / magnitude
            
        # Update position: Position = Position + (Joystick * MaxSpeed * dt)
        self.pos += joy_vector * dt
        self.pos = np.clip(self.pos, -2, 2)
        return self.pos.copy()

    def move(self, pos, vel=None, acc=None):
        """sets cursor to a position"""
        self.pos = pos

    def animate_inputs_duo(self, position1, position2, time, names = [], fps: int = 30, speed=2):
        """
        Animates two dots moving based on two sets of positions over time.

        :param position1: List/array of (x, y) positions for the first dot.
        :param position2: List/array of (x, y) positions for the second dot.
        :param time: List/array of timestamps corresponding to the positions.
        :param fps: Frames per second for the animation render.
        :param speed: Animation playback speed multiplier.
        """
        # Determine how many samples to skip to approximately match the
        # requested animation FPS.
        sample_rate = 1.0 / (time[1] - time[0])
        sample_step = max(1, int(sample_rate / fps))

        # Sample both trajectories using the same timestamps
        trajectory1 = np.asarray(position1)[::sample_step, :2]
        trajectory2 = np.asarray(position2)[::sample_step, :2]

        # Make sure both trajectories have the same number of frames
        num_frames = min(len(trajectory1), len(trajectory2))
        trajectory1 = trajectory1[:num_frames]
        trajectory2 = trajectory2[:num_frames]

        # Set up plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Joystick Dot Simulation")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)

        # Axis limits with margin, considering both trajectories
        all_positions = np.vstack((trajectory1, trajectory2))

        margin = 1.0
        ax.set_xlim(
            np.min(all_positions[:, 0]) - margin,
            np.max(all_positions[:, 0]) + margin
        )
        ax.set_ylim(
            np.min(all_positions[:, 1]) - margin,
            np.max(all_positions[:, 1]) + margin
        )

        # Plot elements
        path_line1, = ax.plot(
            [], [], 'b--', alpha=0.5, label="Path 1"
        )
        dot1, = ax.plot(
            [], [], 'ro', markersize=8, label="Dot 1"
        )

        path_line2, = ax.plot(
            [], [], 'g--', alpha=0.5, label="Path 2"
        )
        dot2, = ax.plot(
            [], [], 'bo', markersize=8, label="Dot 2"
        )

        if names:
            ax.legend(names)
        else:
            ax.legend()

        def init():
            path_line1.set_data([], [])
            dot1.set_data([], [])

            path_line2.set_data([], [])
            dot2.set_data([], [])

            return path_line1, dot1, path_line2, dot2

        def update(frame):
            # Path 1
            path_line1.set_data(
                trajectory1[:frame + 1, 0],
                trajectory1[:frame + 1, 1]
            )
            dot1.set_data(
                [trajectory1[frame, 0]],
                [trajectory1[frame, 1]]
            )

            # Path 2
            path_line2.set_data(
                trajectory2[:frame + 1, 0],
                trajectory2[:frame + 1, 1]
            )
            dot2.set_data(
                [trajectory2[frame, 0]],
                [trajectory2[frame, 1]]
            )

            return path_line1, dot1, path_line2, dot2

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            init_func=init,
            interval=(1000 / speed) / fps,
            blit=True,
            repeat=False
        )

        plt.show()

    def animate_inputs(self, position, time, fps: int = 30, speed = 2):
        """
        Animates the dot moving based on a list of joystick inputs over time.
        
        :param inputs: List of tuples/lists: [(joy_x, joy_y, duration_in_seconds), ...]
        :param fps: Frames per second for the animation render
        """
        dt = 1.0 / fps
        trajectory = []
       
        # Pre-compute positions frame-by-frame
        sampsPerSec = time[-1] / (time[1] - time[0])
        prevTime = 0
        for pos, t in zip(position[::int(sampsPerSec/fps)], time[::int(sampsPerSec/fps)]):
            num_steps = 1 
            for _ in range(num_steps):
                new_pos = np.array([pos[0], pos[1]])
                trajectory.append(new_pos)

        trajectory = np.array(trajectory)

        # Set up plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Joystick Dot Simulation")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)

        # Axis limits with margin
        margin = 1.0
        ax.set_xlim(np.min(trajectory[:, 0]) - margin, np.max(trajectory[:, 0]) + margin)
        ax.set_ylim(np.min(trajectory[:, 1]) - margin, np.max(trajectory[:, 1]) + margin)

        # Plot elements
        path_line, = ax.plot([], [], 'b--', alpha=0.5, label="Path")
        dot, = ax.plot([], [], 'ro', markersize=8, label="Dot")
        ax.legend()

        def init():
            path_line.set_data([], [])
            dot.set_data([], [])
            return path_line, dot

        def update(frame):
            path_line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
            dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
            return path_line, dot

        anim = FuncAnimation(
            fig, update, frames=len(trajectory),
            init_func=init, interval=(1000/speed)/fps, blit=True, repeat=False
        )
        plt.show()

    def animate_joystick(self, velocities, interval=50, save_path=None):
        """
        Animates a 2D joystick given a list of velocity vectors.

        Parameters:
        - velocities: iterable of (vx, vy) tuples/lists representing 2D control input.
        - interval: delay between frames in milliseconds (default: 50ms).
        - save_path: optional string (e.g. 'joystick.gif' or 'joystick.mp4') to export.

        Returns:
        - anim: matplotlib.animation.FuncAnimation object.
        """
        velocities = np.array(velocities[::60])
        
        # Setup plot axis
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')

        # Base circle (joystick boundary limit)
        base_circle = plt.Circle((0, 0), 1.0, color='#333333', fill=False, linewidth=3)
        ax.add_patch(base_circle)
        
        # Crosshairs for visual reference
        ax.axhline(0, color='#cccccc', linestyle='--', linewidth=1)
        ax.axvline(0, color='#cccccc', linestyle='--', linewidth=1)

        # Shaft and knob elements
        shaft, = ax.plot([], [], color='#888888', linewidth=6, zorder=2)
        knob = plt.Circle((0, 0), 0.15, color='#d9534f', zorder=3)
        ax.add_patch(knob)

        # On-screen text for current readout
        readout = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                        fontsize=10, family='monospace', verticalalignment='top')

        def init():
            shaft.set_data([], [])
            knob.center = (0, 0)
            readout.set_text('')
            return shaft, knob, readout

        def update(frame):
            vx, vy = velocities[frame]
            
            # Clamp joystick position inside unit circle boundary
            magnitude = np.hypot(vx, vy)
            if magnitude > 1.0:
                vx, vy = vx / magnitude, vy / magnitude

            shaft.set_data([0, vx], [0, vy])
            knob.center = (vx, vy)
            readout.set_text(f"Vx: {vx:+.2f}\nVy: {vy:+.2f}")
            
            return shaft, knob, readout

        anim = FuncAnimation(
            fig, 
            update, 
            frames=len(velocities), 
            init_func=init, 
            interval=interval, 
            blit=True, 
            repeat=False
        )
        plt.show()

    def inverse_kinematics(self, x, y, t):
        """
        Converts a list of target (X, Y) points into a sequence of joystick commands.
        
        :param points: List of target coordinates [(x1, y1), (x2, y2), ...]
        :return: List of tuples [(joy_x, joy_y, duration), ...]
        """

        joint_pos = np.zeros((len(x), 2))
        prevTime = 0
        current = np.array([x[0], y[0]])
        for i, target in enumerate(zip(x, y)):
            target_pos = np.array(target, dtype=float)
            displacement = target_pos - current
            distance = np.linalg.norm(displacement)

            if distance < 1e-6:
                continue  # Already at target

            if t[i] != 0:
                dt = t[i] - prevTime
                prevTime = t[i]
                joy_x, joy_y = displacement / dt 
                if np.linalg.norm([joy_x, joy_y]) > 1:
                    print(f"can't reach from {current} to {target_pos} in {dt} s, {joy_x}, {joy_y}")
                joint_pos[i, :] = np.array([joy_x, joy_y])
            current = target_pos
        return joint_pos

    def makeJointData(self, pos, vel, acc):
        joint_pos = self.inverse_kinematics(pos)
        print("this shouldn't be getting called, fill the traj")
        return