import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

m = 90
g = 9.81
cwA = 0.47  # cw devided by frontal area A from the equation of air friction for falling objects [m**2]
y0 = 39.045e3   # initial height
v0 = 0.0

rho0 = 1.225  # air density on the ground [kg/m**3]
hs = 8.4e3  # height of earth's atmosphere [m]


def F(y, v):
    """Force as a function of height y and speed v"""
    Fg = -m * g
    rho = rho0 * np.exp(-y / hs)
    Fr = -0.5 * rho * cwA * v * np.abs(v)
    return Fg + Fr


def dgl(t, u):
    y, v = u
    return np.array([v, F(y, v) / m])


def impact(t, u):
    """Event function: returns a sign change at Impact on the ground (y=0)"""
    y, v = u
    return y


impact.terminal = True
u0 = np.array([y0, v0])
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0, events=impact, dense_output=True)
t_s = result.t
y_s, v_s = result.y

t = np.linspace(0, np.max(t_s), 1000)
y, v = result.sol(t)

fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('v [m/s]')
ax1.grid()
ax1.plot(t_s, v_s, '.b')
ax1.plot(t, v, '-b')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('y [m]')
ax2.grid()
ax2.plot(t_s, y_s, '.b')
ax2.plot(t, y, '-b')
ax2.legend()
