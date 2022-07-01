import random

import numpy as np
import typing as t
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

random.seed(0)
class Particle:
    # using r as position vector, holds X,Y coordinates , initialized as 0, 0 two elements
    # v as velocity 0,0 no speed at both direction
    # R is radios for two particles , m is mass for momentum after collision
    # using id to track individual particles.
    # particle initized when created.
    def __init__(self, id=0, r=np.zeros(2), v=np.zeros(2), R=1E-2, m=1, color="blue"):
        # using tuple assignment for defining variables
        self.id, self.r, self.v, self.R, self.m, self.color = id, r, v, R, m, color

    def __repr__(self):
        return f"Particle(id={self.id}, r={self.r}, v={self.v}, R={self.R}, m={self.m}, color={self.color})"


# creating a box for holding particles, the system progresses with time
# dt is time increment, how time passes between each iteration
# Np is number of particles, given values are default.
class Simulation:
    # size of the box in x and y direction to control the height and length of box.
    X = 2
    Y = 2

    def __init__(self, dt=0.6E-2, Np=20):
        self.dt = dt
        self.Np = Np
        # creating particles as a list which creates the class Particle itself
        # it constructs 20 particles by creating itself 20 times according to Np, they are having an id
        self.particles: t.List[Particle] = [Particle(i) for i in range(self.Np)]

    def collision_detection(self):  # it provides the boundary conditions
        ignore_list = []
        # any particle outside the boundary needs to be reflected back inside
        for particle1 in self.particles:
            if particle1 in ignore_list:
                continue

            x, y = particle1.r  # gives the coordinates of the particle
            # if x,y coordinate outside the simulation boundary in either side
            # particle.R provides the boundary(Radios) of the particle itself
            if (x > self.X/2 - particle1.R) or (x < -self.X/2 + particle1.R):
                particle1.v[0] *= -1  # reflecting velocity of X component
            if (y > self.Y/2 - particle1.R) or (y < -self.Y/2 + particle1.R):
                particle1.v[1] *= -1  # 0 and 1 provides the axis
                # reflecting velocity of Y component, here multiplying by -1 makes the direction opposite
                # to compare one particle with other particle, 1 and 2 naming is implemented

            # calculating the difference between radius of two particles
            for particle2 in self.particles:
                # if the particle in the loop is the same particle then ignore it
                if id(particle1) == id(particle2):
                    continue
                m1, m2, r1, r2, v1, v2, = particle1.m, particle2.m, particle1.r, particle2.r, particle1.v, particle2.v
                if np.dot(r1 - r2, r1 - r2) <= (particle1.R + particle2.R)**2: # if two particles are overlapping
                # use the formula of elastic collision to collide overlapping particles
                    v1_new = v1 - 2*m1/(m1+m2) * np.dot(v1 - v2, r1 - r2) / np.dot(r1 - r2, r1 - r2) * (r1 - r2)  # 1st
                    v2_new = v2 - 2*m1/(m1+m2) * np.dot(v2 - v1, r2 - r1) / np.dot(r2 - r1, r2 - r1) * (r2 - r1)  # 2nd
                    particle1.v = v1_new
                    particle2.v = v2_new
                    ignore_list.append(particle2)



    # incrementing particles position based on its speed by using Euler method over dt
    # s = vt tells the incremental change in distance traveled by the particle
    # collision detector is added
    def incrementsim(self):
        self.collision_detection()
        for particle in self.particles:
            particle.r += self.dt * particle.v

    # this function will loop over all particles from the list and returns a list of their position for plotting
    def particle_position(self):
        return [p.r for p in self.particles]

    def particle_colors(self):
        return [p.color for p in self.particles]


    # def __repr__(self):
    #     return f"Simulation(dt={self.dt}, Np={self.Np})"


sim = Simulation()
# creating plot for visualizing the system
# access each particle from simulation and give it position
for particle in sim.particles:  # giving the values of box size
    particle.r = np.random.uniform([-sim.X/2, -sim.Y/2], [sim.X/2, sim.Y/2], size=2)
    # giving the particles velocity to move in an angle for x and y axis
    particle.v = np.array([np.cos(np.pi/4), np.cos(np.pi/4)])


sim.particles[0].color = "red" # for 1 red particle
sim.particles[1].color = "red"
sim.particles[2].color = "red"
sim.particles[3].color = "red"

fig, ax = plt.subplots()
# ax.scatter()
# scatter plot of the particle returns a collection of objects, circles
scatter = ax.scatter([], [])


# creating the animations by updating each change of position
def init_ani():
    ax.set_xlim(-sim.X/2, sim.X/2)
    ax.set_ylim(-sim.Y/2, sim.Y/2)
    return scatter,


def fpsUpdater(frame):
    sim.incrementsim()
    scatter.set_offsets(np.array(sim.particle_position()))
    scatter.set_color(sim.particle_colors())
    return scatter,


ani = FuncAnimation(fig, fpsUpdater, frames=3000, init_func=init_ani, blit=True, interval=1/30, repeat=False)
# ani.save("myani.mp4")


# a = np.array(sim.particle_position())
# ax.scatter(a[:, 0], a[:, 1])
# sim.incrementsim()
# a = np.array(sim.particle_position())
# ax.scatter(a[:, 0], a[:, 1])
