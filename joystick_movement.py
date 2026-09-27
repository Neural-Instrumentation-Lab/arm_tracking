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
        if magnitude > 1.0:
            joy_vector = joy_vector / magnitude
            
        # Update position: Position = Position + (Joystick * MaxSpeed * dt)
        self.pos += joy_vector * dt
        self.pos = np.clip(self.pos, -1, 1)
        return self.pos.copy()

    def move(self, pos, vel=None, acc=None):
        """sets cursor to a position"""
        self.pos = pos

    def animate_inputs(self, position, time, fps: int = 30):
        """
        Animates the dot moving based on a list of joystick inputs over time.
        
        :param inputs: List of tuples/lists: [(joy_x, joy_y, duration_in_seconds), ...]
        :param fps: Frames per second for the animation render
        """
        dt = 1.0 / fps
        trajectory = []
       
        # Pre-compute positions frame-by-frame
        prevTime = 0
        for pos, t in zip(position[::30], time[::30]):
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
            init_func=init, interval=1000/fps, blit=True, repeat=False
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

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize simulator with max speed of 5.0 units/sec at starting position (0, 0)
    sim = JoystickDotSimulator(max_speed=5.0, initial_pos=(0.0, 0.0))

    # 1. Define target waypoints to reach
    waypoints = [
        (5.0, 0.0),
        (5.0, 5.0),
        (-2.0, 3.0),
        (0.0, 0.0)
    ]

    # 2. Run Inverse Kinematics to get required joystick inputs
    joystick_commands = sim.inverse_kinematics(waypoints)

    print("Generated Joystick Commands (joy_x, joy_y, duration):")
    for cmd in joystick_commands:
        print(f"  X: {cmd[0]:6.2f}, Y: {cmd[1]:6.2f}, Time: {cmd[2]:5.2f}s")

    # 3. Animate the dot moving along the calculated joystick inputs
    sim.animate_inputs(joystick_commands, fps=30)