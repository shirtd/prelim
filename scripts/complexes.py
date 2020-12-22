import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from util.data import COLOR

plt.ion()

fig, ax = plt.subplots(1,1)
ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.set_aspect('equal')
ax.axis('off')

r = 2*np.sqrt(2) / 3
P = np.array([[np.cos(2 * np.pi* i / 3), np.sin(2 * np.pi * i / 3)] for i in range(3)])
E = [[0,1], [1,2], [2,0]]
ax.scatter(P[:,0], P[:,1], c='black', zorder=3)

circs = [Circle(p, r) for p in P]
ccoll = PatchCollection(circs, color=COLOR['red'], alpha=0.3, zorder=0)
ax.add_collection(ccoll)

ax.plot(P[E,0], P[E,1], c=COLOR['blue'], zorder=2)

tris = [Polygon(P)]
tcoll = PatchCollection(tris, color=COLOR['blue'], alpha=0.5, zorder=1)
ax.add_collection(tcoll)
