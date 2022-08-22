import numpy as np
import scipy.integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import mpl_toolkits.mplot3d

dim = 3  # dimensions of the box
day = 24 * 60 * 60
year = 365.25 * day
AU = 1.495978707e11
T = 5 * year
dt = 5 * day
G = 6.674e-11

name = ['Sun', 'Merkury', 'Venus', 'Earth',
        'Mars', 'Jupiter', 'Saturn', 'Uranus',
        'Neptune', '9P/Tempel 1', '2010TK7']

color = ['gold', 'darkcyan', 'orange', 'blue',
         'red', 'brown', 'olive', 'green',
         'slateblue', 'black', 'gray']

# Masses of celestial bodies [kg].
m = np.array([1.9885e30, 3.302e23, 48.685e23, 5.9722e24,
              6.4171e23, 1.89813e27, 5.6834e26, 8.68103e25,
              1.02413e26, 7e13, 2e10])

# exact Positions [m] and speeds [m/s] of the celestial bodies
r0 = AU * np.array([
     [-3.241859398499088e-3, -1.331449770492458e-3, -8.441430972210388e-7],
     [-3.824910108111409e-1, -1.955727022061594e-1,  1.892637411059862e-2],
     [ 7.211147749926723e-1,  3.334025180138600e-2, -4.133082682493956e-2],
     [-1.704612905998195e-1,  9.676758607337962e-1, -3.140642423792612e-5],
     [-1.192725241298415e+0,  1.148990485621534e+0,  5.330857335041436e-2],
     [ 3.752622496696632e+0,  3.256207159994215e+0, -9.757709767384923e-2],
     [-8.943506571472968e+0, -3.720744112648929e+0,  4.206153526052092e-1],
     [ 2.003510615298455e+1,  1.205184752774219e+0, -2.550883982941838e-1],
     [ 2.601428919232999e+1, -1.493950125368399e+1, -2.918668092864814e-1],
     [ 3.092919273623052e+0,  6.374849521314798e-1, -4.938170253879825e-1],
     [-3.500888634488231e-1,  7.382457660686845e-1,  9.937175322228885e-2]])

v0 = AU / day * np.array([
     [ 4.270617154820447e-6, -4.648506431568692e-6, -8.469657867642489e-8],
     [ 7.029877499006405e-3, -2.381780604663742e-2, -2.590381459216828e-3],
     [-1.045793358516289e-3,  2.010665107676625e-2,  3.360587977875350e-4],
     [-1.722905169624698e-2, -3.001024883870811e-3,  2.627266603191336e-7],
     [-9.195012836122981e-3, -8.871670960023885e-3,  4.000329706845314e-5],
     [-5.035554237289496e-3,  6.060385207824578e-3,  8.751352649528277e-5],
     [ 1.842572816910875e-3, -5.163338547394546e-3,  1.648327631319252e-5],
     [-2.649722266077889e-4,  3.742642248006496e-3,  1.735555169285604e-5],
     [ 1.542818728068733e-3,  2.740646317666675e-3, -9.236392136662484e-5],
     [ 3.234481019056261e-3,  8.932013115163753e-3,  3.798319697848072e-5],
     [-1.651037000673457e-2, -1.028444949884146e-2,  6.705542557361902e-3]])

# number of celestial bodies
N = len(name)

# Calculate the center of gravity position and speed
# Subtract these from the initial conditions.
r0 -= m @ r0 / np.sum(m)
v0 -= m @ v0 / np.sum(m)


def dgl(t, u):
    r, v = np.split(u, 2)
    r = r.reshape(N, dim)
    a = np.zeros((N, dim))
    for i in range(N):
        for j in range(i):
            dr = r[j] - r[i]
            gr = G / np.linalg.norm(dr) ** 3 * dr
            a[i] += gr * m[j]
            a[j] -= gr * m[i]
    return np.concatenate([v, a.reshape(-1)])


# Fix the state vector at time t=0
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))

t = result.t
r, v = np.split(result.y, 2)

# Convert r and v to a 3-dimensional array:
# 1. Index - celestial bodies
# 2. Index - coordinate direction
# #3. Index - timing
r = r.reshape(N, dim, -1)
v = v.reshape(N, dim, -1)

# Calculate the different energy contributions.
E_kin = 1/2 * m @ np.sum(v * v, axis=1)
E_pot = np.zeros(t.size)
for i in range(N):
    for j in range(i):
        dr = np.linalg.norm(r[i] - r[j], axis=0)
        E_pot -= G * m[i] * m[j] / dr
E = E_pot + E_kin

# calculate the momentum
# interchanges the two axes of an array, view of the swapped array is returned
#
p = m @ v.swapaxes(0, 1)

# Calculate the position of the center of gravity.
rs = m @ r.swapaxes(0, 1) / np.sum(m)

# Calculate the angular momentum.
L = m @ np.cross(r, v, axis=1).swapaxes(0, 1)

fig1 = plt.figure()
fig1.set_tight_layout(True)


# Create an Axes and plot the energy.
ax1 = fig1.add_subplot(2, 2, 1)
ax1.set_title('Energie')
ax1.set_xlabel('t [d]')
ax1.set_ylabel('E [J]')
ax1.grid()
ax1.plot(t / day, E, label='time vs total energy')

# Create an axis and plot the momentum.
ax2 = fig1.add_subplot(2, 2, 2)
ax2.set_title('Impulse')
ax2.set_xlabel('t [d]')
ax2.set_ylabel('\\vec p [kg m / s]')
ax2.grid()
ax2.plot(t / day, p[0, :], '-r', label='p_x')
ax2.plot(t / day, p[1, :], '-b', label='p_y')
ax2.plot(t / day, p[2, :], '-k', label='p_z')
ax2.legend()

# Create an axis and plot angular momentum
ax3 = fig1.add_subplot(2, 2, 3)
ax3.set_title('Drehimpuls')
ax3.set_xlabel('t [d]')
ax3.set_ylabel('\\vec L [kg m² / s]')
ax3.grid()
ax3.plot(t / day, L[0, :], '-r', label='L_x')
ax3.plot(t / day, L[1, :], '-b', label='L_y')
ax3.plot(t / day, L[2, :], '-k', label='L_z')
ax3.legend()

# Excrete an Axes and plot the centroid coordinates.
ax4 = fig1.add_subplot(2, 2, 4)
ax4.set_title('	center of gravity')
ax4.set_xlabel('t [d]')
ax4.set_ylabel('\\vec r_s [m]')
ax4.grid()
ax4.plot(t / day, rs[0, :], '-r', label='r_{s,x}')
ax4.plot(t / day, rs[1, :], '-b', label='r_{s,y}')
ax4.plot(t / day, rs[2, :], '-k', label='r_{s,z}')
ax4.legend()


# Create a figure and a 3D axis for animation
fig2 = plt.figure(figsize=(9, 6))
ax = fig2.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.grid()

# Plot the trajectories for each planet and add them
for i in range(N):
    ax.plot(r[i, 0] / AU, r[i, 1] / AU, r[i, 2] / AU,
            '-', color=color[i], label=name[i])
ax.legend()

# Create a dot plot for each planet in the corresponding color
# save it in the list planet.
planet = []
for i in range(N):
    p, = ax.plot([0], [0], 'o', color=color[i])
    planet.append(p)

# Add a text field to display elapsed time.
text = fig2.text(0.5, 0.95, '')


def update(n):
    for i in range(N):
        planet[i].set_data(np.array([r[i, 0, n] / AU]),
                           np.array([r[i, 1, n] / AU]))
        planet[i].set_3d_properties(r[i, 2, n] / AU)
    text.set_text(f'{t[n] / year:.2f} year')
    return planet + [text]


# Show the graph.
ani = mpl.animation.FuncAnimation(fig2, update, interval=30,
                                  frames=t.size)
plt.show()
