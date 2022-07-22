import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate
from matplotlib.animation import FuncAnimation

# Simulation of a planetary orbit assuming that the sun is resting

day = 24 * 60 * 60 # [s]
year = 365.25 * day # [s]

# astronomical unit [m]
AE = 1.495978707e11

# scale factor for displaying acceleration and speed
scal_a = 20  # [m/s**2]
scal_v = 1e-5  # [m/s]

# Simulation duration T and displayed increment dt [s]
T = 1 * year
dt = 1 * day
M = 1.9885e30  # mass of sun
G = 6.674e-11  # Gravity constant
r0 = np.array([0.0, 29.29e3])  # initial position of planet
v0 = np.array([0.0, 29.29e3])  # initial speed of the planet


def dgl(t, u):
    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


# position vector at time t=0
u0 = np.concatenate((r0, v0))

# equation of motion
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, dense_output=True)

t_s = result.t
r_s, v_s = np.split(result.y, 2)

# calculate the interpolation on a grid
t = np.arange(0, np.max(t_s), dt)
r, v = np.split(result.sol(t), 2)

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('x [AE]')
ax.set_ylabel('y [AE]')
ax.grid()

# plot trajectories of celestial bodies
ax.plot(r_s[0]/ AE, r_s[1]/ AE, '.b')
ax.plot(r[0]/ AE, r[1] / AE, '-b')

# Generate dot plots for the positions of the celestial bodies.
planet, = ax.plot([0], [0], 'o', color='red')
sun, = ax.plot([0], [0], 'o', color='gold')

# Create two arrows for the acceleration vectors
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
arrow_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black', arrowstyle=style)
arrow_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red', arrowstyle=style)

# add the arrow to the axes
ax.add_artist(arrow_a)
ax.add_artist(arrow_v)

# add a text field to indicate the elapsed time
text_t = ax.text(0.01, 0.95, '', color='blue', transform=ax.transAxes)


# Add a text field to indicate the elapsed time
def update(n):
    # Update the position of the celestial body
    planet.set_data(r[:, n] / AE)

    # Calculate the instantaneous acceleration and update it
    u = np.concatenate((r[:, n], v[:, n]))
    u_point = dgl(t[n], u)
    a = np.split(u_point, 2)[1]

    arrow_a.set_positions(r[:, n] / AE, r[:, n] / AE + scal_a * a)
    arrow_v.set_positions(r[:, n] / AE + scal_v * v[:, n])

    # Update the text field for the time
    text_t.set_text(f't = {t[n] / day: 0f} d')
    return planet, arrow_v, arrow_a, text_t


# Create the animation object and start the animation.
ani = mpl.animation.FuncAnimation(fig, update, interval=30, frames=t.size)
plt.show()
