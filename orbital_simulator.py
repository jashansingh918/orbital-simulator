import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Constants
MU = 3.986e14  # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6  # Earth's radius (m)

# Initial conditions
r0 = np.array([R_EARTH + 500e3, 0])  # 500 km altitude (m)
v0 = np.array([0, 7.5e3])  # 7.5 km/s velocity (m/s)
state0 = np.concatenate((r0, v0))

# Time settings
dt = 1.0  # Time step (s)
t_max = 3600  # 1 hour (s)
time = np.arange(0, t_max, dt)

# Compute acceleration
def compute_acceleration(r):
    r_norm = np.linalg.norm(r)
    return -MU * r / r_norm**3

# Simulate orbit (Euler method)
def simulate_orbit(state0, dt, time):
    states = [state0]
    for _ in time[:-1]:
        r = states[-1][:2]
        v = states[-1][2:]
        a = compute_acceleration(r)
        r_new = r + v * dt
        v_new = v + a * dt
        states.append(np.concatenate((r_new, v_new)))
    return np.array(states)

# Run simulation
trajectory = simulate_orbit(state0, dt, time)

# Set up plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1e7, 1e7)
ax.set_ylim(-1e7, 1e7)
ax.set_aspect('equal')
ax.grid(True)

# Draw Earth
earth = plt.Circle((0, 0), R_EARTH, color='blue', alpha=0.5)
ax.add_patch(earth)

# Plot elements
path, = ax.plot([], [], 'r-', lw=1, label='Orbit')
satellite, = ax.plot([], [], 'ko', label='Satellite')

# Animation functions
def init():
    path.set_data([], [])
    satellite.set_data([], [])
    return path, satellite

def update(frame):
    path.set_data(trajectory[:frame, 0], trajectory[:frame, 1])  # Path is a sequence
    satellite.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])  # Satellite as single-point sequence
    return path, satellite

# Create and save animation
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=10)
plt.title("Orbital Trajectory Simulator")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()

# Save as MP4
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, bitrate=1800)
ani.save('orbit.mp4', writer=writer)

print("Simulation complete. Check 'orbit.mp4' in the file explorer.")