"""Simulation of the velocity distribution in a gas. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# box dimensions.
dim = 2

# Number of particles.
N = 200

# Simulation duration T and increment dt [s].
T = 10
dt = 0.01

# Smallest time difference at which collisions than simultaneously be accepted [s].
epsilon = 1e-9

# For each wall, the distance from the coordinate origin
# wall_d and an outward-pointing normal vector wall_n specified.
wall_d = np.array([2.0, 2.0, 2.0, 2.0])
wall_n = np.array([[0, -1.0], [0, 1.0], [-1.0, 0], [1.0, 0]])

# Randomly position the masses in the area
# x= -1.9 ... 1.9 m and y = -1.9 ... 1.9 m.
r0 = 1.9 * (2 * np.random.rand(N, dim) - 1)

# Choose random speeds with magnitude 1 m/s.
v0 = -0.5 + np.random.rand(N, dim)
v0 /= np.linalg.norm(v0, axis=1).reshape(-1, 1)

# All particles get the same mass [kg].
radius = 0.05 * np.ones(N)

# All particles get the same mass [kg].
m = np.ones(N)

# Set the maximum for displaying the histogram
# Speed, the maximum number of particles per bar and the number of bars (n_bins) fixed.
v_max = 3.0
n_max = 50
n_bins = 15

# Create arrays for the simulation result.
t = np.arange(0, T, dt)
print(f"t = {t}")
r = np.empty((t.size, N, dim))
v = np.empty((t.size, N, dim))
r[0] = r0
v[0] = v0


def collision_time(r, v):
    """Returns the time until the next particle collision and the indices of the participating particles. """

    # Create N x N x dim arrays that match the pairwise
    # Location and speed differences included:
    # dr[i, j] is the vector r[i] - r[j]
    # dv[i, j] is the vector v[i] - v[j]
    dr = r.reshape(N, 1, dim) - r
    dv = v.reshape(N, 1, dim) - v

    # Create an N x N array containing the squared absolute value of the
    # Contains vectors from array dv.
    dv2 = np.sum(dv * dv, axis=2)

    # Create an N x N array containing the pairwise sum
    # contains the radii of the particles.
    rad = radius + radius.reshape(N, 1)

    # To determine the time of the collision,
    # form quadratic equation t² + 2 a t + b = 0 be resolved.
    # Only the smaller solution is relevant.
    a = np.sum(dv * dr, axis=2) / dv2
    b = (np.sum(dr * dr, axis=2) - rad ** 2) / dv2
    D = a**2 - b
    t = -a - np.sqrt(D)

    # Find the smallest positive instant of a collision
    t[t <= 0] = np.NaN
    t_min = np.nanmin(t)

    # Find the corresponding particle indices.
    i, j = np.where(np.abs(t - t_min) < epsilon)

    # Select the first half of the indices because each
    # Particle pairing occurs twice.
    i = i[0:i.size // 2]
    j = j[0:j.size // 2]

    # Return time and particle indices. if no collision occurs, then return inf.
    if np.isnan(t_min):
        t_min = np.inf

    return t_min, i, j


def collision_wall(r, v):
    """Returns the time until the next wall collision, the index of the particles and the index of the wall. """

    # Calculate the time of the collision of the particles # one of the walls.
    # The result is an N x number of walls - arrays.
    distance = wall_d - radius.reshape(-1, 1) - r @ wall_n.T
    t = distance / (v @ wall_n.T)

    # Ignore all non-positive tenses.
    t[t <= 0] = np.NaN

    # Ignore all times when the particle moves
    # against the normal vector. Actually
    # this shouldn't happen at all, but due to
    # rounding errors it can happen that a particle
    # is slightly outside a wall.
    t[(v @ wall_n.T) < 0] = np.NaN

    # Find the smallest point in time and give the time and indices back.
    t_min = np.nanmin(t)
    particle, wall = np.where(np.abs(t - t_min) < epsilon)
    return t_min, particle, wall


# Calculate the time until the first collision and partners involved.
dt_particle, particle1, particle2 = collision_time(r[0], v[0])
dt_wall, particle_w, wall = collision_wall(r[0], v[0])
dt_collision = min(dt_particle, dt_wall)

dt_collision_list = []
# Loop over the time steps.

volume = (4 / 3) * np.pi * radius ** 3
VOLUME = volume
temp = 273.15
BLT_CONST = 1.380649e-23

for i in range(1, t.size):
    # Copy the positions from the previous time step.
    r[i] = r[i - 1]
    v[i] = v[i - 1]

    # Time that has already been simulated in this time step..
    t1 = 0

    # Handle all collisions in this one in turn timestep.
    while t1 + dt_collision <= dt:
        # Move all particles forward until collision.
        r[i] += v[i] * dt_collision

        # Collisions between particles among themselves.
        if dt_particle <= dt_wall:
            for k1, k2 in zip(particle1, particle2):
                dr = r[i, k1] - r[i, k2]
                dv = v[i, k1] - v[i, k2]
                er = dr / np.linalg.norm(dr)
                m1 = m[k1]
                m2 = m[k2]
                v1_s = v[i, k1] @ er
                v2_s = v[i, k2] @ er
                #
                #
                positive_dt_collision_list = []
                if len(dt_collision_list) < 1:
                    delt = 0.0001
                    delt_positive = abs(delt)
                    positive_dt_collision_list.append(delt_positive)
                else:
                    for x in range(len(dt_collision_list) - 1):
                        time2 = dt_collision_list[x + 1]
                        time1 = dt_collision_list[x]
                        delt = time2 - time1
                        delt_positive = abs(delt)
                        positive_dt_collision_list.append(delt_positive)
                        print(f"delt= {delt}")
                        print(f"delt positive= {delt_positive}")
                        delt_average = sum(positive_dt_collision_list) / len(positive_dt_collision_list)
                        print(f"delt average= {delt_average}")
                        print(positive_dt_collision_list)
                        # for v1 new
                        mass = m2
                        initial_velocity = v2_s
                        v_rms = np.sqrt((3 * BLT_CONST * temp) / mass)
                        v1 = initial_velocity + v_rms  # it has absorbed radiated energy itself while moving
                        a = v_rms / time1  # an unknown particle with different acceleration about to hit V1 particle
                        v2 = v1 + a * time2  # it is being hit by a particle thus received some energy from that's acceleration
                        v1_s_neu = (1 / 4) * m2 * v2 ** 2 * m2 * (v1 - v2) ** 2 * temp * time1 * (
                                1 / (delt * delt_average * volume * (m1 + m2)))

                        #
                        # for v2 new
                        mass = m1
                        initial_velocity = v1_s
                        v_rms = np.sqrt((3 * BLT_CONST * temp) / mass)
                        v1 = initial_velocity + v_rms  # it has absorbed radiated energy itself while moving
                        a = v_rms / time1  # an unknown particle with different acceleration about to hit V1 particle
                        v2 = v1 + a * time2  # it is being hit by a particle thus received some energy from that's acceleration
                        v2_s_neu = (1 / 4) * m1 * v1 ** 2 * m1 * (v2 - v1) ** 2 * temp * time2 * (
                                1 / (delt * delt_average * volume * (m1 + m2)))
                        #
                        #
                        # v1_s_neu = (2 * m2 * v2_s + (m1 - m2) * v1_s) / (m1 + m2)
                        # v2_s_neu = (2 * m1 * v1_s + (m2 - m1) * v2_s) / (m1 + m2)
                        v[i, k1] += -v1_s * er + v1_s_neu * er
                        v[i, k2] += -v2_s * er + v2_s_neu * er

        # Collisions between particles and walls.
        if dt_particle >= dt_wall:
            for n, w in zip(particle_w, wall):
                v1_s = v[i, n] @ wall_n[w]
                v[i, n] -= 2 * v1_s * wall_n[w]

        # Within this time step was a duration
        # dt_collision already covered.
        t1 += dt_collision

        # Since collisions have taken place, recalculate.
        dt_particle, particle1, particle2 = collision_time(r[i], v[i])
        dt_wall, particle_w, wand = collision_wall(r[i], v[i])
        dt_collision = min(dt_particle, dt_wall)

    # Now find until the end of the current time step (dt).
    # no more collision. We move all particles up
    # forward to the end of the time step and don't have to
    # Check for collisions again.
    r[i] = r[i] + v[i] * (dt - t1)
    dt_collision -= dt - t1
    dt_collision_list.append(dt_collision)

    # Give an information about the progress of the simulation in percent off.
    print(f'{100*i/t.size:.1f} %')

# Create a figure.
fig = plt.figure(figsize=(8, 4))
fig.set_tight_layout(True)

# Create an Axes for the animation of the movement of the particles.
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('particle motion')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_xlim([-2.1, 2.1])
ax1.set_ylim([-2.1, 2.1])
ax1.set_aspect('equal')
ax1.grid()

# Create a circle for each particle.
circles = []
for i in range(N):
    c = mpl.patches.Circle([0, 0], radius[i])
    ax1.add_artist(c)
    circles.append(c)

# Create a second axis for the histogram.
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('speed distribution')
ax2.set_xlabel('|v| [m/s]')
ax2.set_ylabel('Anzahl der Teilchen')
ax2.set_ylim([0, n_max])
ax2.grid()

# Generate the data for the histogram.
hist, bins = np.histogram(np.linalg.norm(v[0], axis=1),
                          bins=n_bins, range=[0, v_max])

# Display the histogram as a bar chart.
bar = ax2.bar(bins[:-1], hist, width=v_max / n_bins,
                 align='edge', edgecolor='white', zorder=2)


def update(n):
    # Update the positions of the particles.
    for i, k in enumerate(circles):
        k.set_center(r[n, i])

    # Calculate the histogram for the current time step.
    hist, bins = np.histogram(np.linalg.norm(v[n], axis=1),
                              bins=n_bins, range=[0, v_max])

    # Update histogram bars.
    for i, p in enumerate(bar):
        p.set_height(hist[i])

    return circles + list(bar)


# Create the animation and start it.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()