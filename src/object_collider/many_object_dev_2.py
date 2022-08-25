"""Simulation of the velocity distribution in a gas. """

import numpy as np
from dataclasses import dataclass
from typing import Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation


@dataclass
class InitConfig:
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

    # All particles get the same mass [kg].
    radius = 0.05 * np.ones(N)

    # All particles get the same mass [kg].
    m = np.ones(N)

# Randomly position the masses in the area
# x= -1.9 ... 1.9 m and y = -1.9 ... 1.9 m.
# r0 = 1.9 * (2 * np.random.rand(N, dim) - 1)

# Choose random speeds with magnitude 1 m/s.
# v0 = -0.5 + np.random.rand(N, dim)
# v0 /= np.linalg.norm(v0, axis=1).reshape(-1, 1)


# Set the maximum for displaying the histogram
# Speed, the maximum number of particles per bar and the number of bars (n_bins) fixed.
@dataclass
class FigureConfig:
    v_max = 3.0
    n_max = 50
    n_bins = 15


# Create arrays for the simulation result.
@dataclass
class Result:
    config: InitConfig
    t: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    r0: Optional[np.ndarray] = None
    v0: Optional[np.ndarray] = None

    def __post_init__(self):
        self.t = np.arange(0, self.config.T, self.config.dt)
        self.r = np.empty((self.t.size, self.config.N, self.config.dim))
        self.v = np.empty((self.t.size, self.config.N, self.config.dim))
        self.r[0] = self.calc_r0()
        self.v[0] = self.calc_v0()

    def calc_r0(self):
        if self.r0:
            return self.r0
        self.r0 = 1.9 * (2 * np.random.rand(self.config.N, self.config.dim) - 1)
        return self.r0

    def calc_v0(self):
        if self.v0:
            return self.v0
        v0 = -0.5 + np.random.rand(self.config.N, self.config.dim)
        v0 /= np.linalg.norm(v0, axis=1).reshape(-1, 1)
        self.v0 = v0
        return v0


@dataclass
class CollisionTimeResult:
    dt_particle: float
    particle1: np.ndarray
    particle2: np.ndarray


def collision_time(result: Result, config: InitConfig):
    """Returns the time until the next particle collision and the indices of the participating particles. """

    # Create N x N x dim arrays that match the pair wise
    # Location and speed differences included:
    # dr[i, j] is the vector r[i] - r[j]
    # dv[i, j] is the vector v[i] - v[j]
    dr = result.r0.reshape(config.N, 1, config.dim) - result.r0
    dv = result.v0.reshape(config.N, 1, config.dim) - result.v0

    # Create an N x N array containing the squared absolute value of the
    # Contains vectors from array dv.
    dv2 = np.sum(dv * dv, axis=2)

    # Create an N x N array containing the pair wise sum
    # contains the radii of the particles.
    rad = config.radius + config.radius.reshape(config.N, 1)

    # To determine the time of the collision,
    # form quadratic equation t² + 2 a t + b = 0 be resolved.
    # Only the smaller solution is relevant.
    a = np.sum(dv * dr, axis=2) / dv2
    b = (np.sum(dr * dr, axis=2) - rad ** 2) / dv2
    D = a**2 - b
    t = -a - np.sqrt(D)
    # print(f"collision time t = {t}")

    # Find the smallest positive instant of a collision
    t[t <= 0] = np.NaN
    t_min = np.nanmin(t)
    # print(f"collision time t_min nan min = {t_min}")
    # Find the corresponding particle indices.
    i, j = np.where(np.abs(t - t_min) < config.epsilon)

    # Select the first half of the indices because each Particle pairing occurs twice.
    i = i[0:i.size // 2]
    j = j[0:j.size // 2]

    # Return time and particle indices. if no collision occurs, then return inf.
    if np.isnan(t_min):
        t_min = np.inf
        # print(f"if loop tmin = {t_min}")

    time_result = CollisionTimeResult(float(t_min), i, j)
    return time_result


@dataclass
class CollisionWallResult:
    dt_wall: float
    particle_w: np.ndarray
    wall: np.ndarray


def collision_wall(result: Result, config: InitConfig):
    """Returns the time until the next wall collision, the index of the particles and the index of the wall. """

    # Calculate the time of the collision of the particles # one of the walls.
    # The result is an N x number of walls - arrays.
    distance = config.wall_d - config.radius.reshape(-1, 1) - result.r0 @ config.wall_n.T
    t = distance / (result.v0 @ config.wall_n.T)

    # Ignore all non-positive tenses.
    t[t <= 0] = np.NaN

    # Ignore all times when the particle moves
    # against the normal vector. Actually
    # this shouldn't happen at all, but due to
    # rounding errors it can happen that a particle
    # is slightly outside a wall.
    t[(result.v0 @ config.wall_n.T) < 0] = np.NaN

    # Find the smallest point in time and give the time and indices back.
    t_min = np.nanmin(t)
    particle, wall = np.where(np.abs(t - t_min) < config.epsilon)
    wall_result = CollisionWallResult(float(t_min), particle, wall)
    return wall_result


conf = InitConfig()
# fig_conf = FigureConfig()
result = Result(config=conf)
col_time_result = collision_time(result=result, config=conf)
col_wal_result = collision_wall(result=result, config=conf)
dt_collision = min(col_time_result.dt_particle, col_wal_result.dt_wall)
print(f"dt= {conf.dt}")
print(f"dt_collision= {dt_collision}")
print(f"dt_particle= {col_time_result.dt_particle}")
print(f"dt_wall= {col_wal_result.dt_wall}")


for i in range(1, result.t.size):
    # Copy the positions from the previous time step.
    result.r[i] = result.r[i - 1]
    result.v[i] = result.v[i - 1]

    # Time that has already been simulated in this time step..
    t1 = 0

    # Handle all collisions in this one in turn timestep.
    while t1 + dt_collision <= conf.dt:
        # Move all particles forward until collision.
        result.r[i] += result.v[i] * dt_collision
        print(f"while t1 = {t1}")
        print(f"while dt_collision = {dt_collision}")

        # Collisions between particles among themselves.
        if col_time_result.dt_particle <= col_wal_result.dt_wall:
            for k1, k2 in zip(col_time_result.particle1, col_time_result.particle2):
                dr = result.r[i, k1] - result.r[i, k2]
                dv = result.v[i, k1] - result.v[i, k2]
                er = dr / np.linalg.norm(dr)
                m1 = conf.m[k1]
                m2 = conf.m[k2]
                v1_s = result.v[i, k1] @ er
                v2_s = result.v[i, k2] @ er
                print(f"v1_s = {v1_s}")
                print(f"v2_s = {v2_s}")
                print(f"m1 = {m1}")
                print(f"m2 = {m2}")

                v1_s_neu = (2 * m2 * v2_s +
                            (m1 - m2) * v1_s) / (m1 + m2)
                v2_s_neu = (2 * m1 * v1_s +
                            (m2 - m1) * v2_s) / (m1 + m2)
                print(f"v1_s_neu = {v1_s_neu}")
                print(f"v2_s_neu = {v2_s_neu}")
                result.v[i, k1] += -v1_s * er + v1_s_neu * er
                result.v[i, k2] += -v2_s * er + v2_s_neu * er
                print(f"v[i, k1] = {result.v[i, k1]}")
                print(f"v[i, k2] = {result.v[i, k2]}")
                print(f"for dt_particle = {col_time_result.dt_particle}")
                print(f"for dt_collision = {dt_collision}")

        # Collisions between particles and walls.
        if col_time_result.dt_particle >= col_wal_result.dt_wall:
            for n, w in zip(col_wal_result.particle_w, col_wal_result.wall):
                v1_s = result.v[i, n] @ conf.wall_n[w]
                result.v[i, n] -= 2 * v1_s * conf.wall_n[w]

        # Within this time step was a duration
        # dt_collision already covered.
        t1 += dt_collision
        print(f"t1 = {t1}")

        # Since collisions have taken place, recalculate.
        col_time_result = collision_time(result=result, config=conf)
        col_wal_result = collision_wall(result=result, config=conf)
        dt_collision = min(col_time_result.dt_particle, col_wal_result.dt_wall)
    print(f"dt_collision = {dt_collision}")

    # Now find until the end of the current time step (dt).
    # no more collision. We move all particles up
    # forward to the end of the time step and don't have to
    # Check for collisions again.
    result.r[i] = result.r[i] + result.v[i] * (conf.dt - t1)
    dt_collision -= conf.dt - t1
    # dt_collision_list.append(dt_collision)
    # print(f"dt_collision in first for loop = {dt_collision}")
    # print(f"dt_collision list in first for loop = {dt_collision_list}")

    # Give an information about the progress of the simulation in percent off.
    print(f'{100*i/result.t.size:.1f} %')

