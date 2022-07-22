import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 360, 500)

y1 = np.sin(np.radians(x))
y2 = np.cos(np.radians(x))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)   # subplot is how the graph looks (nrows, ncols, index)
ax.set_title("sine and cosine")
ax.set_xlabel("Angle")
ax.set_ylabel("function value")
ax.set_xlim(0, 360)  # values in x axis from 0 to 360
ax.set_ylim(-1.1, 1.1)  # values in y axis from -1.1 to 1.1
ax.grid()

ax.plot(x, y1, label="sine")
ax.plot(x, y2, label="cosine")
ax.legend()
plt.show()