import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

day = 24 * 60 * 60
year = 365.25 * day
AU = 1.495978707e11
scal_a = 20  # scaling factor to display accelaration
T = 2 * year
dt = 1 * day
G = 6.674e-11

# mass of the stars
m1 = 2.0e30
m2 = 4.0e29
# initial position of the stars
r0_1 = AU * np.array([0.0, 0.0])
r0_2 = AU * np.array([0.0, 1.0])

# initial velocity
v0_1 = np.array([0.0, 0])
v0_2 = np.array([25e3, 0])


def dgl(t, u):
    r1, r2, v1, v2 = np.split(u, 4)
    a1 = G * m2 / np.linalg.norm(r2 - r1) ** 3 * (r2 - r1)
    a2 = G * m1 / np.linalg.norm(r1 - r2) ** 3 * (r1 - r2)
    return np.concatenate([v1, v2, a1, a2])


# position vector at time t=0
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))
# solve the equation of motion upto time T
# explicitly specify the points in time at which the solution
# of the differential equation is output with the additional argument t_eval
# n the result arrays, the rows represent the vector components and the
# columns represent the time points.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9, t_eval=np.arange(0, T, dt))

t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# calculate different energy contributions
# calculate the potential energy and the kinetic energy for each point in time.
# to determine the square of the absolute value of the speed for each point in time ,
# the squared arrays must be summed over the rows (axis=0) .
E_kin1 = 1/2 * m1 * np.sum(v1 ** 2, axis=0)
E_kin2 = 1/2 * m2 * np.sum(v2 ** 2, axis=0)
E_pot = -G * m1 * m2 / np.linalg.norm(r1 - r2, axis=0)

# The vector of the total momentum
p = m1 * v1 + m2 * v2

# angular momentum
# cross product of two array(or vector) is a scalar value
# angular momentum is calculated via the cross product of the position and the momentum.
# the np.cross function allows to form the cross product of 2-dimensional arrays by substituting zero for the
# third component and returning a scalar quantity as the result, which corresponds to the z
# component of the angular momentum vector.
# cross product has to be formed along the rows (axis=0) , since the columns of the
# arrays represent the individual points in time.
L = m1 * np.cross(r1, v1, axis=0) + m2 * np.cross(r2, v2, axis=0)

fig = plt.figure(figsize=(16, 10))
fig.set_tight_layout(True)

# axis for trijectory of the stars
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlabel('x[AU]')
ax1.set_ylabel('y[AU]')
ax1.set_aspect('equal')
ax1.grid()

# plot trajectories of the stars
ax1.plot(r1[0] / AU, r1[1] / AU, '-r')
ax1.plot(r2[0] / AU, r2[1] / AU, '-b')

# create a dot plot for the position of stars
star1, = ax1.plot([0], [0], 'o', color='red')
star2, = ax1.plot([0], [0], 'o', color='blue')

# two arrows for the acceleration vectors
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
arrow_a1 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red', arrowstyle=style)
arrow_a2 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='blue', arrowstyle=style)
ax1.add_artist(arrow_a1)
ax1.add_artist(arrow_a2)

# axis for the energy
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Energie')
ax2.set_xlabel('t[d]')
ax2.set_ylabel('E[j]')
ax2.grid()
ax2.plot(t / day, E_kin1, '-r', label='E_{kin,1}')
ax2.plot(t / day, E_kin2, '-b', label='E_{kin,2}')
ax2.plot(t / day, E_pot, '-c', label='E_{pot}')
ax2.plot(t / day, E_pot + E_kin1 + E_kin2, '-k', label='E_{total}')
ax2.legend()

# create an axis and plot the angular momentum
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('angular momentum')
ax3.set_xlabel('t[d]')
ax3.set_ylabel('L[kgm²/s]')
ax3.grid()
ax3.plot(t / day, L)

# Generate an Axes and plot the momentum
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('Impulse')
ax4.set_xlabel('t[d]')
ax4.set_ylabel('p[kgm/s]')
ax4.grid()
ax4.plot(t / day, p[0, :], label='p_x')
ax4.plot(t / day, p[1, :], label='p_y')
ax4.legend()

# Make sure that the following lines are no longer Changing y scale.
ax2.set_ylim(auto=False)
ax3.set_ylim(auto=False)
ax4.set_ylim(auto=False)
# Create three black lines representing the current time in the
# plots for energy, momentum and angular momentum.
line_t2, = ax2.plot(0, 0, '-k')
line_t3, = ax3.plot(0, 0, '-k')
line_t4, = ax4.plot(0, 0, '-k')


def update(n):
    # update positions of star
    star1.set_data(r1[:, n] / AU)
    star2.set_data(r2[:, n] / AU)
    # Calculate instantaneous acceleration and update its Vector arrows.
    v_1, v_2, a_1, a_2 = np.split(dgl(t[n], result.y[:, n]), 4)
    arrow_a1.set_positions(r1[:, n] / AU, r1[:, n] / AU + scal_a * a_1)
    arrow_a2.set_positions(r2[:, n] / AU, r2[:, n] / AU + scal_a * a_2)
    # Plot the time on the other three charts.
    t_act = t[n] / day
    line_t2.set_data([[t_act, t_act], ax2.get_ylim()])
    line_t3.set_data([[t_act, t_act], ax3.get_ylim()])
    line_t4.set_data([[t_act, t_act], ax4.get_ylim()])
    return (star1, star2, arrow_a1, arrow_a2,
            line_t2, line_t3, line_t4)


ani = mpl.animation.FuncAnimation(fig, update, interval=30, frames=t.size, blit=True)
plt.show()
