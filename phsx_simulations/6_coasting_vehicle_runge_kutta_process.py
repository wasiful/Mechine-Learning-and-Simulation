import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

t_max = 200  # duration of time to be simulated[s]
dt = 0.2  # time step size[s]
m = 15.0  # vehicle mass[kg]
b = 2.5  # coefficient of the friction [kg/m]
x0 = 0  # starting location[m]
v0 = 10.0  # starting velocity[m/s]


def F(v):
    """ force as function of speed v """
    return -b * v * np.abs(v)  # abs takes the "|-v|= v" absolute of v


t = np.arange(0, t_max, dt)
x = np.empty(t.size)
v = np.empty(t.size)


# The first argument is the time t and second argument the state vector u, dgl exactly needs these two arguments
# state vector u is broken down into its components, the location x and the velocity v .
# Then the expression evaluated for the right-hand side of the  differential equation and returned as an array.
def dgl(t, u):
    x, v = u
    return np.array([v, F(v) / m])


# Fix the state vector at time t=0
u0 = np.array([x0, v0])
# Solve the equation of motion in the time interval
# from t=0 to t=t_max.
# we give it an array of velocity and force dgl, a list of time , an array of initial position x0 and velocity v0 in u0
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0)  # here integrating makes the curve smoother

# Issue the status message and distribute the result to appropriate arrays.
# result.t contains a 1-dimensional array of the times at which the solution has been calculated
# result.y is a 2-dimensional array whose Rows are the components of the state vector and
# its columns are the points in time represent.
print(result.message)
t = result.t
x, v = result.y


fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('v [m/s]')
ax1.grid()
ax1.plot(t, v0 / (1 + v0 * b / m * t), '-b', label='analytical')
ax1.plot(t, v, '.r', label='simulated')     # this plot is using the result to create the '.' graph
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('x [m]')
ax2.grid()
ax2.plot(t, m / b * np.log(1 + v0 * b / m * t), '-b', label='analytical')
ax2.plot(t, x, '.r', label='simulated')     # this plot is using the result to create the '.' graph
ax2.legend()
