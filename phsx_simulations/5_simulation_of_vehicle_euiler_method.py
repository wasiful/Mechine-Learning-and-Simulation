import numpy as np
import matplotlib.pyplot as plt

t_max = 20  # duration of time to be simulated[s]
dt = 0.2  # time step size[s]
m = 15.0  # vehicle mass[kg]
b = 2.5  # coefficient of the friction [kg/m]
x0 = 0  # starting location[m]
v0 = 10.0  # starting velocity[m/s]


def F(v):
    """ force as function of speed v """
    return -b * v * np.abs(v)  # abs takes the "|-v|= v" absolute of v


# create array for the simulation result
t = np.arange(0, t_max, dt)     # from 0 to 20s  it takes 0.2 seconds in each step [0.2, 0.4, ......19.8]
# print(t)
x = np.empty(t.size)  # list created with number of elements in t=time list, 100 randomly taken elements
v = np.empty(t.size)

# setting the initial condition for the loop
x[0] = v0
v[0] = v0

# loop of the simulation updates x[0] -> x[0+1] -> x[1+1].... and v[0] as well
for i in range(t.size - 1):
    x[i+1] = x[i] + v[i] * dt
    v[i+1] = v[i] + F(v[i]) / m * dt

# create a figure
fig = plt.figure(figsize=(9, 4))  # creates the plot figur, the height and width of the graph
fig.set_tight_layout(True)   # remove any commission between the labels in the plot

# plot the velocity vs time graph
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel('t[s]')
ax1.set_ylabel('v[m/s]')
ax1.grid()
ax1.plot(t, v0/(1 + v0 * b / m * t), '-b', label='analytical',)
ax1.plot(t, v, '.r', label='simulated')
ax1.legend()

# plot the space-time diagram
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('t[s]')
ax2.set_ylabel('x[m]')
ax2.grid()
ax2.plot(t, m / b * np.log(1 + v0 * b / m * t), '-b', label='analytical')
ax2.plot(t, x, '.r', label='simulated')
ax2.legend()

# display the graph
plt.show()
