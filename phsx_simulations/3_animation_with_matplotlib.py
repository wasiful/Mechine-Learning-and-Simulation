import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# 𝑢(𝑥, 𝑡) = cos(𝑘𝑥 − 𝜔𝑡) is going to be animated, K and X is given, function of (x,t) is calculated
x = np.linspace(0, 20, 500)
omega = 1.0  # angular frequency
k = 1.0  # wave number
delta_t = 0.02  # time steps, how fast the sine wave moves

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("position x")
ax.set_ylabel("u(x, t)")
ax.text(0.0, 1.05, '')
ax.grid()

plot, = ax.plot(x, 0 * x)  # the separator line in the graph
text = ax.text(0.0, 1.05, '')   # the gap between the beginning and ending grid


# n is the value of the frames assigned by FuncAnimation
def update(n):
    """Calculate the wave at the nth time step and
    update the corresponding graphic elements."""
    t = n * delta_t     # Calculate the function at time t
    u = np.cos(k * x - omega * t)   # formula
    plot.set_ydata(u)   # Update plot.
    text.set_text(f't = {t:5.1f}')      # Return a tuple containing the graphic elements
    return plot, text


ani = mpl.animation.FuncAnimation(fig, update, interval=60, blit=True)
# ani.save('animation_with_matplotlib.mp4')
plt.show()
