import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
L1, L2 = 1, 1  # Length of pendulum arms
m1, m2 = 1, 1  # Masses
g = 9.81  # Acceleration due to gravity

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

# Initial conditions: [theta1, omega1, theta2, omega2]
initial_state = [np.pi / 2, 0, np.pi, 0]

# Time span for the simulation
t_span = [0, 20]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solving the ODE
sol = solve_ivp(derivs, t_span, initial_state, t_eval=t_eval)

# Extracting the results
theta1 = sol.y[0]
theta2 = sol.y[2]

# Plotting the animation
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    x = [0, L1 * np.sin(theta1[frame]), L2 * np.sin(theta2[frame])]
    y = [0, -L1 * np.cos(theta1[frame]), -L2 * np.cos(theta2[frame])]
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True)
plt.show()
