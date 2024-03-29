"""Simulation eines Doppelsternsystems im Schwerpunktssystem. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate


day = 24 * 60 * 60
year = 365.25 * day

AU = 1.495978707e11

scal_a = 20
T = 2 * year
dt = 1 * day
G = 6.674e-11
m1 = 2.0e30
m2 = 4.0e29

r0_1 = AU * np.array([0.0, 0.0])
r0_2 = AU * np.array([0.0, 1.0])

v0_1 = np.array([0.0, 0])
v0_2 = np.array([25e3, 0])

# Calculate the center of gravity position and speed and
# Subtract these from the initial conditions.
rs0 = (m1 * r0_1 + m2 * r0_2) / (m1 + m2)
vs0 = (m1 * v0_1 + m2 * v0_2) / (m1 + m2)
r0_1 -= rs0
r0_2 -= rs0
v0_1 -= vs0
v0_2 -= vs0


def dgl(t, u):
    r1, r2, v1, v2 = np.split(u, 4)
    a1 = G * m2 / np.linalg.norm(r2 - r1)**3 * (r2 - r1)
    a2 = G * m1 / np.linalg.norm(r1 - r2)**3 * (r1 - r2)
    return np.concatenate([v1, v2, a1, a2])


u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))

t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

E_kin1 = 1/2 * m1 * np.sum(v1 ** 2, axis=0)
E_kin2 = 1/2 * m2 * np.sum(v2 ** 2, axis=0)
E_pot = - G * m1 * m2 / np.linalg.norm(r1 - r2, axis=0)

p = m1 * v1 + m2 * v2

# Calculate the position of the center of gravity.
rs = (m1 * r1 + m2 * r2) / (m1 + m2)

L = m1 * np.cross(r1, v1, axis=0) + m2 * np.cross(r2, v2, axis=0)

fig = plt.figure(figsize=(10, 7))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlabel('$x$ [AE]')
ax1.set_ylabel('$y$ [AE]')
ax1.set_aspect('equal')
ax1.grid()
ax1.plot(r1[0] / AU, r1[1] / AU, '-r')
ax1.plot(r2[0] / AU, r2[1] / AU, '-b')

star1, = ax1.plot([0], [0], 'o', color='red')
star2, = ax1.plot([0], [0], 'o', color='blue')


style = mpl.patches.ArrowStyle.Simple(head_length=6,
                                      head_width=3)
arrow_a1 = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                       color='red',
                                       arrowstyle=style)
arrow_a2 = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                       color='blue',
                                       arrowstyle=style)

ax1.add_artist(arrow_a1)
ax1.add_artist(arrow_a2)

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Energy')
ax2.set_xlabel('$t$ [d]')
ax2.set_ylabel('$E$ [J]')
ax2.grid()
ax2.plot(t / day, E_kin1, '-r', label='$E_{kin,1}$')
ax2.plot(t / day, E_kin2, '-b', label='$E_{kin,2}$')
ax2.plot(t / day, E_pot, '-c', label='$E_{pot}$')
ax2.plot(t / day, E_pot + E_kin1 + E_kin2,
         '-k', label='$E_{ges}$')
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('angular momentum')
ax3.set_xlabel('$t$ [d]')
ax3.set_ylabel('$L$ [kg m² / s]')
ax3.grid()
ax3.plot(t / day, L)

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('main focus')
ax4.set_xlabel('t [d]')
ax4.set_ylabel('$r_s$ [mm]')
ax4.grid()
ax4.plot(t / day, 1e3 * rs[0, :], label='$r_{s,x}$')
ax4.plot(t / day, 1e3 * rs[1, :], label='$r_{s,y}$')
ax4.legend()

ax2.set_ylim(auto=False)
ax3.set_ylim(auto=False)
ax4.set_ylim(auto=False)

line_t2, = ax2.plot(0, 0, '-k')
line_t3, = ax3.plot(0, 0, '-k')
line_t4, = ax4.plot(0, 0, '-k')


def update(n):

    star1.set_data(r1[:, n] / AU)
    star2.set_data(r2[:, n] / AU)

    v_1, v_2, a_1, a_2 = np.split(dgl(t[n], result.y[:, n]), 4)
    arrow_a1.set_positions(r1[:, n] / AU,
                           r1[:, n] / AU + scal_a * a_1)
    arrow_a2.set_positions(r2[:, n] / AU,
                           r2[:, n] / AU + scal_a * a_2)


    t_akt = t[n] / day
    line_t2.set_data([[t_akt, t_akt], ax2.get_ylim()])
    line_t3.set_data([[t_akt, t_akt], ax3.get_ylim()])
    line_t4.set_data([[t_akt, t_akt], ax4.get_ylim()])

    return (star1, star2, arrow_a1, arrow_a2,
            line_t2, line_t3, line_t4)


ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
fig.set_tight_layout(True)
plt.show()
