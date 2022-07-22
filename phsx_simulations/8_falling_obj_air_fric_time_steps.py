import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# giving properly measured data changes the maximum speed reach during the free fall then previous simulation
m = 90.0
g = 9.81
cwA = 0.47
y0 = 39.045e3
v0 = 0.0

# Air pressure [kg/m³] as a function of altitude
h_mess = 1e3 * np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11.02, 15, 20.06, 25, 32.16, 40])

# Generate an interpolation function for the air density
rho_mess = np.array([1.225, 1.112, 1.007, 0.909, 0.819, 0.736,
                     0.660, 0.590, 0.526, 0.467, 0.414, 0.364,
                     0.195, 0.0880, 0.0401, 0.0132, 0.004])


# Generate an interpolation function for the air density.
fill = (rho_mess[0], rho_mess[-1])
rho = scipy.interpolate.interp1d(h_mess, rho_mess, kind='cubic', bounds_error=False, fill_value=fill)


def F(y, v):
    Fg = -m * g
    Fr = -0.5 * rho(y) * cwA * v * np.abs(v)    # using rho instead of the barometric height formula
    return Fg + Fr


def dgl(t, u):
    y, v = u
    return np.array([v, F(y, v) / m])


def aufprall(t, u):
    y, v = u
    return y


aufprall.terminal = True


u0 = np.array([y0, v0])

result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0, events=aufprall, dense_output=True)
t_s = result.t
y_s, v_s = result.y

t = np.linspace(0, np.max(t_s), 1000)
y, v = result.sol(t)

fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('v [m/s]')
ax1.grid()
ax1.plot(t_s, v_s, '.b')  # calculated plot
ax1.plot(t, v, '-b')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('y [m]')
ax2.grid()
ax2.plot(t_s, y_s, '.b')
ax2.plot(t, y, '-b')

plt.show()