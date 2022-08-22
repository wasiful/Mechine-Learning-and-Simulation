import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation


dim = 2  # Number of room dimensions.
T = 100  # Simulation duration T
dt = 0.005  # time step or increment dt [s].

# initial positions of particles [m]  (x,y)
# limit to six identical particles that can move in a rectangular field
# store the initial positions of the particles in a 6 × 2 array:
r0 = np.array([[-1.0, 0.0],  [0.5, 0.0], [0.45, -0.05],
               [0.45, 0.05], [0.55, -0.05], [0.55, 0.05]])

# Number of particles.
N = r0.shape[0]

# Initial speeds [m/s].
# all initial speeds is zero.
# Only the first particle should move to the right with a speed of 3 m/s.
v = np.zeros((N, dim))
# the particle will move 3 meters each second in x direction
v[0] = np.array([3.0, 0.0])  # velocity of first object is only along x axis

# radii of the individual particles [m]
# create an array that contains the radius for each individual sphere.
radius = 0.03 * np.ones(N)
m = 0.2 * np.ones(N)

# Hessian normal form , in which a normal vector and the
# distance from the coordinate origin are specified for each plane (or straight line)
# For each wall, the distance from the coordinate origin
# wall_d and an outward-pointing normal vector wall_n specified.
# The first two items set the limits at x = ±1.2 m
# and the following items set the limits at y = ±0.6 m.
wall_d = np.array([1.2, 1.2, 0.6, 0.6])  # upper boundary of the wall
wall_n = np.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])

# we set a tolerance that becomes relevant when two particles at the same time collide.
# Smallest time difference at which collisions than simultaneously be accepted [s].
epsilon = 1e-9

# Create arrays for the simulation result.
#  we create arrays in which the simulation result is stored.
# At this point, we refrain from storing the speed at each individual point in time, since the speeds only change at the
# moment of the impact anyway .
# store the position
# vectors of the particles for each simulation time step of duration dt in 3-dimensional array.
# first dimension specifies the individual time steps that
# second dimension the individual particles, and the third dimension is the coordinate direction.
# At the same time we store the initial state in the first time step.
# Loop over the time steps.
t = np.arange(0, T, dt)
r = np.empty((t.size, N, dim))

# We copy the starting location into the first row of the result arrays and calculate the
# Time of collision of the two spheres using the collision function.
r[0] = r0


# calculate moment of collision
# If the balls are at the locations vector-r1 or vector-r2
# at the time t = 0 , then if they move uniformly at a later time t they will
# be at the locations r1 + t vector-v1 or r2 + t vector-v2 .
# The balls collide when
# the distance between the two ball centers is equal to the sum of the radii.
# ∣(𝑟1⃑ + 𝑡𝑣1⃑ ) − (𝑟2⃑ + 𝑡𝑣2⃑ )∣ = (𝑅1 + 𝑅2 )**2
# 𝑡 2 + 2𝑡𝑎 + 𝑏 = 0
# 𝑡 = −𝑎 − √𝐷, 𝐷 = 𝑎**2 − 𝑏
# If the balls collide, then the equation has two solutions, of which only the
# solution with the smaller value of t is relevant. The second solution would
# describe the situation where the balls have penetrated and then just touched
# their surfaces again.
# if the discriminant ÿ is non-negative. We implement this solution in a Python function.
# we want the position and velocity vectors
# of particles passed at a given time and obtained as a result, like
# long it takes until the next collision and which of the particles are involved.
def collision_time(r, v):
    """Indicates the time until the next particle collision and the indices of the participating particles. """
    # pairwise difference of all position and velocity vectors
    # since these appear in the solution of the quadratic equation
    # Create N x N x dim arrays containing the pairwise differences
    # Location and speed differences included:
    # dr[i, j] is the vector r[i] - r[j]
    # dv[i, j] is the vector v[i] - v[j]
    dr = r.reshape(N, 1, dim) - r
    dv = v.reshape(N, 1, dim) - v

    # now calculate the times of all conceivable particle collisions
    # with the @ operator, we now use pointwise multiplication
    # and sum over the dimension containing the coordinate directions.

    # Create an N x N array containing the squared absolute value of the
    # Contains vectors from array dv.
    dv2 = np.sum(dv * dv, axis=2)
    # Create an N x N array containing the pairwise sum
    # contains the radii of the particles.
    rad = radius + radius.reshape(N, 1)
    # To determine the time of the collision,
    # form quadratic equation  t² + 2 a t + b = 0
    # be resolved. Only the smaller solution is relevant.
    a = np.sum(dv * dr, axis=2) / dv2
    b = (np.sum(dr * dr, axis=2) - rad ** 2) / dv2
    D = a ** 2 - b

    # To check whether a number is finite or not,
    # there are a number of functions in NumPy such as np.isnan, np.isfinite and np.isinf.

    t = -a - np.sqrt(D)
    # Find the smallest positive instant of a collision.

    # we get an N×N array. The entry t[i, j] indicates the time duration
    # until the i-th and j-th particles collide, or it contains NaN if the particles
    # not collide.
    # select the smallest positive time from the array t .
    # we set all entries less than or equal to zero to NaN and then use
    # the function np.nanmin, which returns the smallest element of the array, where
    # NaN entries are skipped.
    t[t <= 0] = np.NaN
    t_min = np.nanmin(t)

    # use the np.where function to search for all indices where a collision occurs
    # Time t_min takes place. In order to compensate for rounding errors, it is checked
    # whether the absolute value of the difference is smaller than the tolerance epsilon .
    # Find the corresponding particle indices.
    i, j = np.where(np.abs(t - t_min) < epsilon)
    print(f"i = {i}")
    print(f"j = {j}\n")

    # Since all collisions appear twice in the array t , we only consider those
    # first half of the indices.
    # Select the first half of the indices because each
    # Particle pairing occurs twice.
    i = i[0:i.size // 2]
    j = j[0:j.size // 2]

    # replace the time NaN with infinity in the result, since this is the
    # Simplified comparisons in the main program, and give the time as well as the lists
    # of the associated collision partner.
    # Return time and particle indices. if
    # no collision occurs, then return inf.
    if np.isnan(t_min):
        t_min = np.inf

    return t_min, i, j


def boundary_of_sim(r, v):
    """Returns the time until the next wall collision, the index of the particles and the index of the wall."""
    # Calculate the time of the collision of the particles
    # one of the walls. The result is an N x number of walls - arrays.
    distance = wall_d - radius.reshape(-1, 1) - r @ wall_n.T
    t = distance / (v @ wall_n.T)  # wall_n is the position(x,y) of the wall
    # Ignore all non-positive times.
    t[t <= 0] = np.NaN
    # Ignore all times when the particle moves
    # against the normal vector but due to
    # rounding errors it can happen that a particle is slightly outside a wall.
    t[(v @ wall_n.T) < 0] = np.NaN
    # Find the smallest point in time and give the time and the indices back.
    t_min = np.nanmin(t)
    particle, wall = np.where(np.abs(t - t_min) < epsilon)
    return t_min, particle, wall


# Calculate the time until the first collision and the
# partners involved.
dt_particle, particle1, particle2 = collision_time(r[0], v)
dt_wall, particle_w, wall = boundary_of_sim(r[0], v)
dt_collision = min(dt_particle, dt_wall)


for i in range(1, t.size):
    # start the loop that iterates over all times to be displayed
    # we have already copied the initial conditions into the first line of the result arrays,
    # we start with index 1. At the beginning of each loop iteration we copy the
    # Positions from the previous time step
    # Copy the positions from the previous time step.
    r[i] = r[i - 1]

    # Time that has already been simulated in this time step.
    t1 = 0

    # Handle all collisions in this one in turn timestep
    while t1 + dt_collision <= dt:
        # Move all particles forward until collision.
        r[i] += v * dt_collision

        # Collisions between particles among themselves.
        if dt_particle <= dt_wall:
            # a = (1, 2, 3); b = (4, 5, 6) ; zip(a, b) = ((1,4),(2, 5),(3,6))
            # so k1 in particle1 & k2 in particle2
            # here particle1 and 2 comes from collision-time() function
            # holding 2d-(x,y) value containing array, x = velocity(x,y) ;  y=r[0]= position of the particle(x,y)
            for k1, k2 in zip(particle1, particle2):
                # we need to determine the velocities after the collision.
                # calculate the respective velocity component perpendicular to the plane of contact the balls position.
                # multiplying the dot product of the vector
                # by the forms a unit vector that connects the two sphere centers
                # i=increase of time with time step from collision time function
                # r[i, k1]= new position of the particle at time t=i and k1= velocity, previous position
                dr = r[i, k1] - r[i, k2]
                dv = v[k1] - v[k2]
                print(f"dr{dr}")
                print(f"v1{v[k1]}")
                print(f"v2{v[k2]}")
                print(f"dv{dv}\n")
                er = dr / np.linalg.norm(dr)
                m1 = m[k1]
                m2 = m[k2]
                v1_s = v[k1] @ er
                v2_s = v[k2] @ er

                # 𝑚1 𝑣1 + 𝑚2 𝑣2 = 𝑚1 𝑣1′ + 𝑚2 𝑣2′
                # We now calculate the vertical velocity components after the collision
                v1_s_new = (2 * m2 * v2_s + (m1 -m2) * v1_s) / (m1 + m2)
                v2_s_new = (2 * m1 * v1_s + (m2 - m1) * v2_s) / (m1 + m2)

                # We get the respective speed after the collision by
                # subtracting the original vertical component from the original velocity
                # and adding the new vertical component.
                v[k1] += (v1_s_new - v1_s) * er
                v[k2] += (v2_s_new - v2_s) * er

        # Collisions between particles and walls.
        if dt_particle >= dt_wall:
            for n, w in zip(particle_w, wall):
                v1_s = v[n] @ wall_n[w]
                v[n] -= 2 * v1_s * wall_n[w]

        # Within this time step Duration dt_collision already handled.
        t1 += dt_collision

        # Since collisions have taken place, we need these recalculate.
        dt_particle, particle1, particle2 = collision_time(r[i], v)
        dt_wall, particle_w, wall = boundary_of_sim(r[i], v)
        dt_collision = min(dt_particle, dt_wall)

    # Now find until the end of the current time step (dt)
    # no more collision. We move all particles
    # forward to the end of the time step and don't have to
    # Check for collisions again.

    r[i] += v * (dt - t1)
    dt_collision -= dt - t1

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-0.6, 0.6])
ax.set_aspect('equal')
ax.grid()


# Create a circle for each particle.
circle = []
for i in range(N):
    c = mpl.patches.Circle([0, 0], radius[i])
    ax.add_artist(c)
    circle.append(c)


def update(n):
    for i in range(N):
        circle[i].set_center(r[n, i])
    return circle


ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()








