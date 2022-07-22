import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# simulation of a flying ball with air friction
# scaling factors for velocity vector and acceleration vector
scal_v = 0.1
scal_a = 0.1
m = 2.7e-3
cwA = 0.45 * math.pi * 20e-3 ** 2
r0 = np.array([0, 1.1])     # starting position[m]
alpha = math.radians(40.0)  # launching angle [rad]
v0 = 20     # amount of dropping speed
g = 9.81
rho = 1.225     # air density[kg/m**3]

v0 = np.array([v0 * math.cos(alpha), v0 * math.sin(alpha)])     # calculating initial velocity vector


def F(r, v):
    """calculating force as velocity and position vector"""
    Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
    Fg = m * g * np.array([0, -1])
    return Fg + Fr


def dgl(t, u):
    r, v = np.split(u, 2)
    return np.concatenate([v, F(r, v) / m])


def impact(t, u):
    """for sign change at impact on the ground"""
    r, v = np.split(u, 2)
    return r[1]


impact.terminal = True
u0 = np.concatenate((r0, v0))  # merges two list into one
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0, events=impact, dense_output=True)

t_s = result.t
r_s, v_s = np.split(result.y, 2)

# calculate interpolation on fine grid
t = np.linspace(0, np.max(t_s), 1000)   # start, stop, amount of points
r, v = np.split(result.sol(t), 2)

fig = plt.figure(figsize=(9, 4))

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_aspect('equal')
ax.grid()
ax.plot(r_s[0], r_s[1], '.b')
ax.plot(r[0], r[1], '-b')


# Create a dot plot for the position of the ball
ball, = ax.plot([0], [0], 'o', color='red', zorder=4)

# Create arrows for speed and acceleration.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)

arrow_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red', arrowstyle=style, zorder=3)
arrow_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black', arrowstyle=style, zorder=3)

# add the graphic objects to thw axis
ax.add_artist(arrow_v)
ax.add_artist(arrow_a)

# Create text boxes to display the current one
# Velocity and Acceleration Amount.
text_t = ax.text(2.1, 1.5, '', color='blue')
text_v = ax.text(2.1, 1.1, '', color='red')
text_a = ax.text(2.1, 0.7, '', color='black')


def update(n):
    ball.set_data(r[:, n])    # updates the position of the ball
    a = F(r[:, n], v[:, n])/m  # calculates the continuous acceleration

    # Update the arrows for speed and acceleration
    arrow_v.set_positions(r[:, n], r[:, n] + scal_v * v[:, n])
    arrow_a.set_positions(r[:, n], r[:, n] + scal_a * a)

    # Update the text fields
    text_t.set_text(f't = {t[n]:.2f} s')
    text_v.set_text(f'v = {np.linalg.norm(v[:, n]):.1f} m/s')
    text_a.set_text(f'a = {np.linalg.norm(a):.1f} m/s²')

    return ball, arrow_v, arrow_a, text_v, text_a, text_t


# Create the animation object and start the animation
ani = mpl.animation.FuncAnimation(fig, update, interval=30, frames=t.size, blit=True)
plt.show()
