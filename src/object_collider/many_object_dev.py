from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation


# Global Variables
config_path = Path(__file__).parent.parent.parent.joinpath('configs')
INIT_CONFIG = OmegaConf.load(config_path.joinpath("init.yaml"))
FIG_CONFIG = OmegaConf.load(config_path.joinpath("visual.yaml"))
RADIUS = 0.05 * np.ones(INIT_CONFIG.N)
WALL_D = np.array(INIT_CONFIG.wall_d)
WALL_N = np.array(INIT_CONFIG.wall_n)


@dataclass
class SimulationResult:
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    r0: np.ndarray
    v0: np.ndarray

    def __post_init__(self):
        self.r[0] = self.r0
        self.v[0] = self.v0


def init_simulation_result(no_of_particles, sim_duration, dt, dim) -> SimulationResult:
    """
    Initialize the result holder of Particle simulator
    :param no_of_particles: No. of particle for simulation
    :param sim_duration: Duration of simulation
    :param dt: Duration interval
    :param dim: Box dimensions from config
    :return: A Result Holder type
    """
    t = np.arange(0, sim_duration, dt)
    r = np.empty((t.size, no_of_particles, dim))
    v = np.empty((t.size, no_of_particles, dim))
    v0 = -0.5 + np.random.rand(no_of_particles, dim)
    v0 /= np.linalg.norm(v0, axis=1).reshape(-1, 1)
    r0 = 1.9 * (2 * np.random.rand(no_of_particles, dim) - 1)
    return SimulationResult(t, r, v, r0, v0)


def collision_time(r, v, no_of_particles, dim, epsilon, radius) -> tuple:
    """
    Returns the time until the next particle collision and the indices of the participating particles.
    :param r:
    :param v:
    :param no_of_particles: No. of particle for simulation
    :param dim: Box dimensions from config
    :param epsilon: Smallest time difference at which collisions than simultaneously be accepted [s].
    :param radius: All particles get the same mass [kg].
    :return: Tuple of resultant values
    """

    # Create N x N x dim arrays that match the pair wise location and speed differences included:
    # dr[i, j] is the vector r[i] - r[j]
    # dv[i, j] is the vector v[i] - v[j]
    dr = r.reshape(no_of_particles, 1, dim) - r
    dv = v.reshape(no_of_particles, 1, dim) - v

    # Create an N x N array containing the squared absolute value of the Contains vectors from array dv.
    dv2 = np.sum(dv * dv, axis=2)

    # Create an N x N array containing the pair wise sum contains the radii of the particles.
    rad = radius + radius.reshape(no_of_particles, 1)

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
    i, j = np.where(np.abs(t - t_min) < epsilon)

    # Select the first half of the indices because each Particle pairing occurs twice.
    i = i[0:i.size // 2]
    j = j[0:j.size // 2]

    # Return time and particle indices. if no collision occurs, then return inf.
    if np.isnan(t_min):
        t_min = np.inf
        # print(f"if loop tmin = {t_min}")

    return t_min, i, j


def collision_wall(r, v, wall_d, wall_n, radius, epsilon) -> tuple:
    """
    Returns the time until the next wall collision, the index of the particles and the index of the wall.
    :param r:
    :param v:
    :param wall_d: For each wall, the distance from the coordinate origin
    :param wall_n: an outward-pointing normal vector wall_n specified
    :param radius: All particles get the same mass [kg].
    :param epsilon: Smallest time difference at which collisions than simultaneously be accepted [s].
    :return: Tuple of resultant values
    """

    # Calculate the time of the collision of the particles # one of the walls.
    # The result is an N x number of walls - arrays.
    distance = wall_d - radius.reshape(-1, 1) - r @ wall_n.T
    t = distance / (v @ wall_n.T)

    # Ignore all non-positive tenses.
    t[t <= 0] = np.NaN

    # Ignore all times when the particle moves against the normal vector. Actually
    # this shouldn't happen at all, but due to rounding errors it can happen that
    # a particle is slightly outside a wall.
    t[(v @ wall_n.T) < 0] = np.NaN

    # Find the smallest point in time and give the time and indices back.
    t_min = np.nanmin(t)
    particle, wall = np.where(np.abs(t - t_min) < epsilon)
    return t_min, particle, wall


def run_sim(result: SimulationResult, config: DictConfig, wall_d, wall_n, radius):
    dt_particle, particle1, particle2 = collision_time(
        result.r[0], result.v[0], config.N, config.dim, config.epsilon, radius
    )
    dt_wall, particle_w, wall = collision_wall(result.r[0], result.v[0], wall_d, wall_n, radius, config.epsilon)
    dt_collision = min(dt_particle, dt_wall)
    # All particles get the same mass [kg].
    m = np.ones(config.N)

    print(f"dt= {config.dt}")
    print(f"dt_collision= {dt_collision}")
    print(f"dt_particle= {dt_particle}")
    print(f"dt_wall= {dt_wall}")

    for i in range(1, result.t.size):
        # Copy the positions from the previous time step.
        result.r[i] = result.r[i - 1]
        result.v[i] = result.v[i - 1]

        # Time that has already been simulated in this time step..
        t1 = 0

        # Handle all collisions in this one in turn timestep.
        while t1 + dt_collision <= config.dt:
            # Move all particles forward until collision.
            result.r[i] += result.v[i] * dt_collision
            print(f"while t1 = {t1}")
            print(f"while dt_collision = {dt_collision}")

            # Collisions between particles among themselves.
            if dt_particle <= dt_wall:
                for k1, k2 in zip(particle1, particle2):
                    dr = result.r[i, k1] - result.r[i, k2]
                    dv = result.v[i, k1] - result.v[i, k2]
                    er = dr / np.linalg.norm(dr)
                    m1 = m[k1]
                    m2 = m[k2]
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
                    print(f"for dt_particle = {dt_particle}")
                    print(f"for dt_collision = {dt_collision}")

            # Collisions between particles and walls.
            if dt_particle >= dt_wall:
                for n, w in zip(particle_w, wall):
                    v1_s = result.v[i, n] @ wall_n[w]
                    result.v[i, n] -= 2 * v1_s * wall_n[w]

            # Within this time step was a duration
            # dt_collision already covered.
            t1 += dt_collision
            print(f"t1 = {t1}")

            # Since collisions have taken place, recalculate.
            dt_particle, particle1, particle2 = collision_time(
                r=result.r, v=result.v, no_of_particles=config.N,
                dim=config.dim, epsilon=config.epsilon, radius=radius
            )
            dt_wall, particle_w, wand = collision_wall(
                r=result.r, v=result.v,
                wall_d=wall_d, wall_n=wall_n,
                radius=radius, epsilon=config.epsilon
            )
            dt_collision = min(dt_particle, dt_wall)
        # print(f"dt_collision = {dt_collision}")

        # Now find until the end of the current time step (dt).
        # no more collision. We move all particles up
        # forward to the end of the time step and don't have to
        # Check for collisions again.
        result.r[i] = result.r[i] + result.v[i] * (config.dt - t1)
        dt_collision -= config.dt - t1
        # dt_collision_list.append(dt_collision)
        # print(f"dt_collision in first for loop = {dt_collision}")
        # print(f"dt_collision list in first for loop = {dt_collision_list}")

        # Give an information about the progress of the simulation in percent off.
        print(f'{100*i/result.t.size:.1f} %')
    return result


def run_animation(result: SimulationResult, config: DictConfig, fig_config: DictConfig, radius: np.ndarray):
    fig = plt.figure(figsize=(8, 4))
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('particle motion')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_xlim([-2.1, 2.1])
    ax1.set_ylim([-2.1, 2.1])
    ax1.set_aspect('equal')
    ax1.grid()

    circles = []
    for i in range(config.N):
        c = mpl.patches.Circle([0, 0], radius[i])
        ax1.add_artist(c)
        circles.append(c)

    # Create a second axis for the histogram.
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('speed distribution')
    ax2.set_xlabel('|v| [m/s]')
    ax2.set_ylabel('number of particles')
    ax2.set_ylim([0, fig_config.n_max])
    ax2.grid()

    # Generate the data for the histogram.
    hist, bins = np.histogram(
        np.linalg.norm(fig_config.v[0], axis=1),
        bins=fig_config.n_bins, range=[0, fig_config.v_max]
    )
    # Display the histogram as a bar chart.
    bar = ax2.bar(
        bins[:-1], hist, width=fig_config.v_max / fig_config.n_bins,
        align='edge', edgecolor='white', zorder=2
    )

    def update(n):
        # Update the positions of the particles.
        for i, k in enumerate(circles):
            k.set_center(result.r[n, i])

        # Calculate the histogram for the current time step.
        hist, bins = np.histogram(np.linalg.norm(result.v[n], axis=1),
                                  bins=fig_config.n_bins, range=[0, fig_config.v_max])

        # Update histogram bars.
        for i, p in enumerate(bar):
            p.set_height(hist[i])

        return circles + list(bar)

    # Create the animation and start it.
    ani = mpl.animation.FuncAnimation(
        fig, update, interval=30,
        frames=result.t.size, blit=True
    )
    plt.show()


def main():
    result = init_simulation_result(
        INIT_CONFIG.N,
        INIT_CONFIG.T,
        INIT_CONFIG.dt,
        INIT_CONFIG.dim
    )
    result = run_sim(result=result, config=INIT_CONFIG, wall_d=WALL_D, wall_n=WALL_N, radius=RADIUS)
    run_animation(result=result, config=INIT_CONFIG, fig_config=FIG_CONFIG, radius=RADIUS)


if __name__ == "__main__":
    main()