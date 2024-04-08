import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
L1, L2 = 1, 1  # Length of pendulum arms
m1, m2 = 1, 1  # Masses
g = 9.81  # Acceleration due to gravity
perturbation = 0.3  # Initial perturbation for the angles

def derivs(t, state):

    """
    Calculate the derivatives of the states.

    state[0] = theta1
    state[1] = omega1
    state[2] = theta2
    state[3] = omega2
    """

    theta1, omega1, theta2, omega2 = state

    delta_theta = theta2 - theta1
    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta_theta) ** 2
    denominator2 = (L2 / L1) * denominator1

    dydx = np.zeros_like(state)
    dydx[0] = omega1
    dydx[1] = (m2 * L1 * omega1 ** 2 * np.sin(delta_theta) * np.cos(delta_theta)
               + m2 * g * np.sin(theta2) * np.cos(delta_theta)
               + m2 * L2 * omega2 ** 2 * np.sin(delta_theta)
               - (m1 + m2) * g * np.sin(theta1)) / denominator1
    dydx[2] = omega2
    dydx[3] = (-L2 / L1) * (m2 * L1 * omega1 ** 2 * np.sin(delta_theta) * np.cos(delta_theta)
                            + m2 * g * np.sin(theta2) * np.cos(delta_theta)
                            + m2 * L2 * omega2 ** 2 * np.sin(delta_theta)
                            - (m1 + m2) * g * np.sin(theta1)) / denominator2

    return dydx

# Number of pendulums
num_pendulums = 30

# Array of initial conditions: [theta1, omega1, theta2, omega2]
initial_states = np.zeros((num_pendulums, 4))
initial_states[:, 0] = np.pi / 2 + np.random.uniform(-perturbation, perturbation, num_pendulums)  # Initial angle for theta1

# Time span for the simulation
t_span = [0, 20]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE for each set of initial conditions
solutions = []
for i in range(num_pendulums):
    sol = solve_ivp(derivs, t_span, initial_states[i], t_eval=t_eval)
    solutions.append(sol.y)

# Plotting the animation
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid()

lines = [ax.plot([], [], 'o-', lw=2)[0] for _ in range(num_pendulums)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    for i, line in enumerate(lines):
        x = [0, L1 * np.sin(solutions[i][0, frame]), L1 * np.sin(solutions[i][0, frame]) + L2 * np.sin(solutions[i][2, frame])]
        y = [0, -L1 * np.cos(solutions[i][0, frame]), -L1 * np.cos(solutions[i][0, frame]) - L2 * np.cos(solutions[i][2, frame])]
        line.set_data(x, y)
    return lines

# Increase the number of frames and update interval for smoother animation
frames = len(t_eval)
interval = 1000 * t_span[1] / frames  # milliseconds

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval, repeat=False)
plt.show()
