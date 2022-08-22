#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML


# In[2]:


dim = 2
# number of particles
N = 10
# Simulation duration T and increment dt [s]
T = 10
dt = 0.02
# Spring constant at impact [N/m]
D = 5e3


# In[3]:


# Randomly position the masses in the area
# x=0.5 ... 1.5 and y = 0.5 ... 1.5 [m].
#  random.rand(R, C) = a matrix of R number of rows and C number of columns(10, 2)
r0 = 0.5 + np.random.rand(N, dim)
r0


# In[4]:


# Choose random speeds in range
# vx = -0.5 ... 0.5 and vy = -0.5 ... 0.5 [m/s]
v0 = -0.5 + np.random.rand(N, dim)
v0


# In[5]:


# Choose random radii in the range from 0.02 to 0.04 [m] for 10 balls
radius = 0.02 + 0.02 * np.random.rand(N)
radius


# In[6]:


# Choose random masses in berevon from 0.2 to 2.0 [kg], 1-d array
m = 0.2 + 1.8 * np.random.rand(N)
m


# In[7]:


def dgl(t, u):
    r, v = np.split(u, 2)
    r = r.reshape(N, dim)
    a = np.zeros((N, dim))

    print(f"r {r}")
    print(f"v {v}")
    print(f"u {u}")
    print(f"umax {max(u)}")
    print(f"umin {min(u)}")
    print(f"a {a}")
    for i in range(N):
        for j in range(i):
            # Calculate the distance between the centers.
            dr = np.linalg.norm(r[i] - r[j])  # norm = length = resultant
            # Calculate the penetration depth.
            # only in x axis
            # max takes the highest from the list of given values
            dist = max(radius[i] + radius[j] - dr, 0)

            # The force should be proportional to the penetration depth.
            F = D * dist
            er = (r[i] - r[j]) / dr
            a[i] += F / m[i] * er
            a[j] -= F / m[j] * er
    return np.concatenate([v, a.reshape(-1)])


# In[8]:


# Fix the state vector at time t=0.
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))
u0


# In[9]:


result = scipy.integrate.solve_ivp(dgl, [0, T], u0, max_step=dt,
                                   t_eval=np.arange(0, T, dt))
result


# In[10]:


t = result.t
t


# In[11]:


r, v = np.split(result.y, 2)
r
v


# In[22]:


# Convert r and v to a 3-dimensional array:
# 1. Index - particle
# 2. Index - coordinate direction
# 3. Index - timing
r = r.reshape(N, dim, -1)
v = v.reshape(N, dim, -1)
print("0")
print(r)
print("0")
print(v)
print("0")


# In[13]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_aspect('equal')
ax.grid()


# In[14]:


# Add the graphic objects to the Axes.
ball = []
for i in range(N):
    c = mpl.patches.Circle([0, 0], radius[i])
    ax.add_artist(c)
    ball.append(c)
ball


# In[15]:


def update(n):
    for i in range(N):
        ball[i].set_center(r[i, :, n])
    return ball


# In[16]:


ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
# plt.show()


# In[17]:


# HTML(ani.to_html5_video())


# In[18]:


get_ipython().system('jupyter nbconvert 15_collision.ipynb --to python --output myfile.py')

