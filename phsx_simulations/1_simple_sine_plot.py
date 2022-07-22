import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 360, 500)  # 500 points taken from 0 to 360 degree
y = np.sin(np.radians(x))  # radian converts degrees into radian values, and we take sine of those values

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y)
plt.show()
