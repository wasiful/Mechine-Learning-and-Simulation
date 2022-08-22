import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Newtonian Equation of motion solved with solve_ivp.
# Simulation duration T and increment dt [s].
T = 8
dt = 0.2

# Spring constant at impact [N/m].
# the larger the spring constant, the closer
# larger the spring constant hardens ball and the error in the energy gets bigger.
# stiff problem, actual push occurs on very short timescales
D = 5e3

# kg
m1 = 1.0
m2 = 2.0

# radios of two particles m
R1 = 0.1
R2 = 0.3

# initial position m
r0_1 = np.array([-2.0, 0.1])  # the first object comes from far and hits the second object
r0_2 = np.array([0.0, 0.0])  # the second object is steady at the  middle

# initial speed m/s
v0_1 = np.array([1.0, 0.0])  # the first object has velocity in only x direction
v0_2 = np.array([0.0, 0.0])  # the second object is motionless


def dgl(t, u):
    # after collision the resultant is the direction of the ball
    # r1 is the vector for the first ball from the 0,0 to its position and r2 is for second
    # v1 is the direction and speed of the first ball
    r1, r2, v1, v2 = np.split(u, 4)  #what is u here

    # Calculate the distance between the centers. norm is the length of a vector
    # sqrt(x1^2, x2^2, x3^2...) if vector-x = (x1, x2, x3, ....)
    # here norm is the distance from suppose a point A(1,5) to the point in the graph B(4, 3)
    dr = np.linalg.norm(r1 - r2)

    # Calculate how far the balls penetrated each other.
    # the commission happened in x-axis so y=0
    # subtract the distance between the centers from the sum of the two radii.
    # Using the max function , ensured that penetration depth
    # dist is zero if the spheres are not touching.
    dist = max(R1 + R2 - dr, 0)

    # The force acts as soon as the surfaces touch.
    # Proportionality constant D N/m was defined
    F = D * dist

    # Compute vectors of acceleration.
    # Acceleration vector is parallel to the
    # Line connecting the two sphere centers.
    er = (r1 - r2) / dr
    a1 = F / m1 * er
    a2 = -F / m2 * er
    # Return the time derivative of the state vector.
    return np.concatenate([v1, v2, a1, a2])


# Fix the state vector at time t=0.
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

# Solve the equation of motion up to time T.
# max_step, which specifies the maximum time step size. Used to indicate the time of collision
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, max_step=dt, t_eval=np.arange(0, T, dt))
print(result)

t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# Calculate energy and total momentum before and after collision and
# print these values.
# kinetic energy and the total momentum before and
# output after the impact process.
E0 = 1 / 2 * (m1 * np.sum(v1[:, 0] ** 2) + m2 * np.sum(v2[:, 0] ** 2))
E1 = 1 / 2 * (m1 * np.sum(v1[:, -1] ** 2) + m2 * np.sum(v2[:, -1] ** 2))
p0 = m1 * v1[:, 0] + m2 * v2[:, 0]
p1 = m1 * v1[:, -1] + m2 * v2[:, -1]

print(f'                       before          afterwards')
print(f'Energies [J]:          {E0:8.5f}      {E1:8.5f}')
print(f'Impulse x [kg m / s]:  {p0[0]:8.5f}      {p1[0]:8.5f}')
print(f'Impulse y [kg m / s]:  {p0[1]:8.5f}      {p1[1]:8.5f}')

# Create a figure and an axis with labels.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-2.0, 2.0])
ax.set_ylim([-1.5, 1.5])
ax.set_aspect('equal')
ax.grid()

# Create the line plots for the trajectory.
track1, = ax.plot([0], [0], '-r', zorder=4)
track2, = ax.plot([0], [0], '-b', zorder=3)

# Create two circles to represent the bodies. at the (0,0) position
circle1 = mpl.patches.Circle([0, 0], R1, color='red', zorder=4)
circle2 = mpl.patches.Circle([0, 0], R2, color='blue', zorder=3)
ax.add_artist(circle1)  # red circle
ax.add_artist(circle2)  # blue circle


def update(n):
    # Update the position of the two bodies.
    circle1.set_center(r1[:, n])
    circle2.set_center(r2[:, n])
    # Plot the trajectory up to the current time.
    track1.set_data(r1[0, :n], r1[1, :n])
    track2.set_data(r2[0, :n], r2[1, :n])
    return circle1, circle2, track1, track2


ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()