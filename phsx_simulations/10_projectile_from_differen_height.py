import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

m = 14.5
cwA = 0.45 * math.pi * 8e-2 ** 2  # Product of c_w value and frontal area[m^2]
g = 9.81
rho = 1.225  # air density
r0 = np.array([0, 0, 10.0])  # starting position[m]
alpha = math.radians(42.0)  # launch angle
v0 = 150.0

# calculate the initial velocity vector[m/s]
v0 = np.array([v0 * math.cos(alpha), 0, v0 * math.sin(alpha)])
theta = math.radians(49.4)  # latitude
omega = 7.29e-5 * np.array([0, math.cos(theta), math.sin(theta)])  # angular velocity vector


def F(r, v):
    """Vector of force as a function of position(r) and velocity"""
    Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
    Fg = m * g * np.array([0, 0, -1])
    Fc = -2 * m * np.cross(omega, v)  #
    return Fg + Fr + Fc


def dgl(t, u):
    r, v = np.split(u, 2)
    return np.concatenate([v, F(r, v) / m])


def impact(t, u):
    """event function returns a sign change at impact on the ground"""
    r, v = np.split(u, 2)
    return r[2]


impact.terminal = True  # stop simulation upon impact
u0 = np.concatenate((r0, v0))  # state vector(position, velocity) at time t=0

# solve the  equation of motion until it hits the ground
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0, events=impact, dense_output=True)

t_s = result.t
r_s, v_s = np.split(result.y, 2)

# calculate interpolation on a fine grid
t = np.linspace(0, np.max(t_s), 1000)
r, v = np.split(result.sol(t), 2)

fig = plt.figure(figsize=(16, 10))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(labelbottom=False)
ax1.set_ylabel('Z[m]')
ax1.set_aspect('equal')
ax1.grid()
ax1.plot(r_s[0], r_s[2], '.b')
ax1.plot(r[0], r[2], '-b')

ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
ax2.set_xlabel('x [m]')
ax2.set_xlabel('y [m]')
ax2.grid()
ax2.plot(r_s[0], r_s[2], '.b')
ax2.plot(r[0], r[1], '-b')
plt.show()