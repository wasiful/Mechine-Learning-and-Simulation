import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate


"""two hard spheres with given
radii R1 and R2 , which are located at the locations vector-r1 and vector-r2 at the time t = 0 and, meet one another
with the constant velocities vector-v1 and vector-v2 without
friction to move. The position vectors each refer to the center of the sphere."""

dim = 2
# number of particles
N = 10
# Simulation duration T and increment dt [s]
T = 10
dt = 0.02
# Spring constant at impact [N/m]
D = 5e3

# Randomly position the masses in the area
# x=0.5 ... 1.5 and y = 0.5 ... 1.5 [m].
#  random.rand(R, C) = a matrix of R number of rows and C number of columns
# the first column is plotted as x and second as y in the graph
r0 = 0.5 + np.random.rand(N, dim)
print(f"r0 {r0}")

# Choose random speeds in range
# vx = -0.5 ... 0.5 and vy = -0.5 ... 0.5 [m/s]
# - is used to randomly take both negative and positive x, y values so the particles move in all direction
v0 = -0.5 + np.random.rand(N, dim)

# Choose random radii in the range from 0.02 to 0.04 [m] for 10 balls
# radius is also randomly taken within a range
radius = 0.02 + 0.02 * np.random.rand(N)

# Choose random masses in berevon from 0.2 to 2.0 [kg], 1-d array
m = 0.2 + 1.8 * np.random.rand(N)

print(f" m {m}")


def dgl(t, u):
    # u is an 1d-array which holds 40 random values
    #  as we have 10 balls in simulation 10-x, 10-y so we need 20 values
    # as we have a position and a velocity so we need 20 for r and 20 for v
    # 40 values are splitted to 20  and 20 set of values and assigned in position r and velocity v
    r, v = np.split(u, 2)
    r = r.reshape(N, dim) # reshaping turns thoes 20 values into a 10 row, 2 column 2d-array as dim=2
    a = np.zeros((N, dim))
    print(f"r {r}")
    print(f"v {v}")
    print(f"u {u}")
    print(f"umax {max(u)}")
    print(f"umin {min(u)}")
    print(f"a {a}")

    for i in range(N):
        for j in range(i):
            # Calculate the distance between the centers.
            dr = np.linalg.norm(r[i] - r[j])  # norm = length = resultant
            # Calculate the penetration depth.
            # only in x axis
            # max takes the highest from the list of given values
            dist = max(radius[i] + radius[j] - dr, 0)

            # The force should be proportional to the penetration depth.
            F = D * dist
            er = (r[i] - r[j]) / dr
            a[i] += F / m[i] * er
            a[j] -= F / m[j] * er
    return np.concatenate([v, a.reshape(-1)])


# Fix the state vector at time t=0.
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))
print(f" u0 {m}")
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, max_step=dt,
                                   t_eval=np.arange(0, T, dt))
print(f" result {m}")
t = result.t
r, v = np.split(result.y, 2)


# Convert r and v to a 3-dimensional array:
# 1. Index - particle
# 2. Index - coordinate direction
# 3. Index - timing
r = r.reshape(N, dim, -1)
v = v.reshape(N, dim, -1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_aspect('equal')
ax.grid()

# Add the graphic objects to the Axes.
ball = []
for i in range(N):
    c = mpl.patches.Circle([0, 0], radius[i])
    ax.add_artist(c)
    ball.append(c)


def update(n):
    for i in range(N):
        ball[i].set_center(r[i, :, n])
    return ball


ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()