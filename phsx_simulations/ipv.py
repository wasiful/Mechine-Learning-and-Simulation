from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


def f(t, r):
    x, y = r  # 2d array r
    fx = np.cos(y)
    fy = np.sin(x)
    return fx, fy # returns in tuples
# solve ipv turns every thing into array


# x(t) = y(t) = a, x(0) = y(0) = 1
sol = integrate.solve_ivp(f, (0, 10), (1, 1), t_eval=np.linspace(0, 10, 100))
print(sol)
x, y = sol.y  # sol.y is a 2d array
print(x)
print(y)
plt.plot(x, y)
plt.show()
