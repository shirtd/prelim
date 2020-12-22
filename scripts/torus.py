import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from util.data import *
import dionysus as dio
from util.plot import plot_diagram
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from util.plot import SurfacePlot, SurfaceCut
import os

from util.diagram import *
from util.bottleneck import *
from util.plot import *
from load_surf import VIEW, tup_d, edges_to_path

class FeaturePlot:
    def __init__(self, chain, X, Y, G, chain_color=COLOR['red'], rad=0.03):
        C = [[grid_coord(v, len(X)) for v in s] for s in chain]
        paths = edges_to_path(C)
        chains = [np.array([[X[a,b], Y[a,b], G[a,b]] for a,b in p]) for p in paths]
        self.pts = np.vstack([c[:-1] for c in chains])
        self.centroid = self.pts.sum(axis=0) / len(self.pts)
        self.radius = max(la.norm(self.centroid - p) for p in self.pts)
        self.elements = [self.plot(c, rad, chain_color) for c in chains]
    def plot(self, curve, radius, color):
        c = plot3d(curve[:,0], curve[:,1], curve[:,2])
        c.parent.parent.filter.radius = radius
        c.actor.property.lighting = False
        c.actor.property.color = color
        return c
    def remove(self):
        for e in self.elements:
            e.parent.parent.parent.parent.remove()

r1, r2 = 0.5, 1
phi, theta = np.mgrid[-np.pi:np.pi:65j, -np.pi:np.pi:65j]

x = (r1 * np.cos(phi) + r2) * np.cos(theta)
y = (r1 * np.cos(phi) + r2) * np.sin(theta)
z = r1 * np.sin(phi)

fun = abs(y) #+ 0.01 * (np.random.rand(*y.shape) - 0.5)

_l = fun.max() - fun.min()
_d = _l / 4

# CUTS = [fun.min(), fun.min() + _d, fun.min() + 2*_d, fun.min() + 3*_d, fun.max()]
CUTS = [0, 0.35, 0.7, 1.05, 1.5]

SURF_ARGS = {   'A' : {'min' : CUTS[0], 'max' : CUTS[1],    'color' : COLOR['green'],   'opacity' : 0.5},
                'B' : {'min' : CUTS[1], 'max' : CUTS[2],    'color' : COLOR['blue'],    'opacity' : 0.5},
                'C' : {'min' : CUTS[2], 'max' : CUTS[3],    'color' : COLOR['purple'],  'opacity' : 0.5},
                'D' : {'min' : CUTS[3], 'max' : CUTS[4],    'color' : COLOR['yellow'],  'opacity' : 0.5}}

CONT_ARGS = {   'A_c' : {'scalar' : [CUTS[1]], 'color' : COLOR['green']},
                'B_c' : {'scalar' : [CUTS[2]], 'color' : COLOR['blue']},
                'C_c' : {'scalar' : [CUTS[3]], 'color' : COLOR['purple']}}

CMAP = [(COLOR['green'],    (CUTS[0], CUTS[1])),
        (COLOR['blue'],     (CUTS[1], CUTS[2])),
        (COLOR['purple'],   (CUTS[2], CUTS[3])),
        (COLOR['yellow'],   (CUTS[3], CUTS[4]))]


DIR = os.path.join('figures', 'torus')
if not os.path.exists(DIR):
    os.mkdir(DIR)

SAVE = False

surf = mlab.mesh(x, y, z, scalars=fun, opacity=0.3)
gcf = mlab.gcf()
scene = gcf.scene
scene.background = (1,1,1)
mlab.view(0.0, 0.0, 8.587096893655424, np.array([0.005, 0., 0.]))
scene.camera.parallel_scale = 2.222504218218719

SCUTS = {k : SurfaceCut(surf.parent, k, **v) for k,v in SURF_ARGS.items()}
surf.visible = False

if SAVE:
    mlab.savefig(os.path.join(DIR, 'dump.png'), size=(10,10))
    mlab.savefig(os.path.join(DIR, 'dump.png'), size=(10,10))

F = dio.fill_freudenthal(fun)

N, M = fun.shape

def tov(i, j):
    return j * N + i

vmap = {s[0] : s.data for s in F if s.dimension() == 0}

for i in range(N):
    u, v, w = tov(i, M-1), tov(i, 0), tov((i+1) % N, M-1)
    fu, fv, fw = vmap[u], vmap[v], vmap[w]
    F.add(dio.Simplex([u, v], max(fu, fv)))
    F.add(dio.Simplex([v, w], max(fv, fw)))
    F.add(dio.Simplex([u, v, w], max((fu, fv, fw))))
    if i > 0:
        t = tov(i-1, 0)
        F.add(dio.Simplex([t, u, v], max((vmap[t], fu, fv))))

for j in range(M):
    u, v, w = tov(N-1, j), tov(0, j), tov(N-1, (j+1) % M)
    fu, fv, fw = vmap[u], vmap[v], vmap[w]
    F.add(dio.Simplex([u, v], max(fu, fv)))
    if j < M-1:
        F.add(dio.Simplex([v, w], max(fv, fw)))
        F.add(dio.Simplex([u, v, w], max((fu, fv, fw))))
    if j > 0:
        t = tov(0, j-1)
        F.add(dio.Simplex([t, u, v], max((vmap[t], fu, fv))))

u, v, w = tov(0,0), tov(0,M-1), tov(N-1,0)
F.add(dio.Simplex([u, v, w], max((vmap[u], vmap[v], vmap[w]))))
# F = dio.Filtration([s for s in F if s.data < CUTS[1]])
F.sort()

_F = DioFilt(F, fun, relative=False)
dgms = phcol(_F)

plt.ion()
fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(11,4))
ax[0].set_title(r"$\mathrm{H}_0$")
ax[1].set_title(r"$\mathrm{H}_1$")
ax[2].set_title(r"$\mathrm{H}_2$")
plt.tight_layout()

LIM = max(max(c[1]) for c in CMAP) * 1.2

for i, axis in enumerate(ax):
    plot_diagram(axis, dgms.as_list(i), CMAP, alpha=0.75, size=10, zorder=1)

if __name__ == "__main__":
    f_plt = []
    p_plt = []

    if SAVE:
        plt.savefig(os.path.join(DIR, 'dgm.png'), dpi=300)

    p = dgms[1][0]
    p_plt.append(ax[1].scatter(p.birth, p.death, s=15, marker='D', color=COLOR['black'], zorder=3)) # get_color([p.birth, p.death], CMAP)
    p_plt.append(ax[1].text(p.birth-0.07, p.death-0.07, r"$(%0.2f,%0.2f)$" % (p.birth, p.death), fontsize=7))
    chain = dgms.get_chain(p)
    f_plt.append(FeaturePlot(chain, x, y, z, COLOR['black'], rad=0.04))

    if SAVE:
        plt.savefig(os.path.join(DIR, 'dgm_A1.png'), dpi=300)

    pp = dgms[1][1]
    p_plt.append(ax[1].scatter(pp.birth, LIM, s=15, marker='D', color=COLOR['green'], zorder=3)) # get_color([p.birth, p.death], CMAP)
    p_plt.append(ax[1].text(pp.birth-0.07, LIM-0.07, r"$(%0.2f,\infty)$" % pp.birth, fontsize=7))
    cchain = dgms.get_chain(pp)
    f_plt.append(FeaturePlot(cchain, x, y, z, COLOR['green'], rad=0.05))

    if SAVE:
        plt.savefig(os.path.join(DIR, 'dgm_A2.png'), dpi=300)

    ppp = dgms[1][2]
    p_plt.append(ax[1].scatter(ppp.birth, LIM, s=15, marker='D', color=COLOR['blue'], zorder=3)) # get_color([p.birth, p.death], CMAP)
    p_plt.append(ax[1].text(ppp.birth-0.07, LIM-0.07, r"$(%0.2f,\infty)$" % ppp.birth, fontsize=7))
    ccchain = dgms.get_chain(ppp)
    f_plt.append(FeaturePlot(ccchain, x, y, z, COLOR['blue'], rad=0.05))

    if SAVE:
        plt.savefig(os.path.join(DIR, 'dgm_B1.png'), dpi=300)

    if SAVE:
        plt.savefig(os.path.join(DIR, 'dgm_D0.png'), dpi=300)

    q = dgms[2][0]
    p_plt.append(ax[2].scatter(q.birth, LIM, s=15, marker='D', color=COLOR['yellow'], zorder=3)) # get_color([p.birth, p.death], CMAP)
    p_plt.append(ax[2].text(q.birth-0.07, LIM-0.07, r"$(%0.2f,\infty)$" % q.birth, fontsize=7))
    qchain = dgms.get_chain(q)

    T = [list(s) for s in qchain]
    t = mlab.triangular_mesh(x.flatten(), y.flatten(), z.flatten(), np.array(T), color=COLOR['yellow'])
    t.actor.property.lighting = False
    t.actor.property.representation = 'wireframe'
    t.actor.property.line_width = 0.1

    if SAVE:
        mlab.savefig(os.path.join(DIR, 'surf_D1.png'), size=(2000,2000), magnification=1)
        plt.savefig(os.path.join(DIR, 'dgm_D1.png'), dpi=300)

        t.visible = False
        mlab.savefig(os.path.join(DIR, 'surf_D0.png'), size=(2000,2000), magnification=1)

        SCUTS['C']['visible'] = False
        SCUTS['D']['visible'] = False
        mlab.savefig(os.path.join(DIR, 'surf_B1.png'), size=(2000,2000), magnification=1)

        for e in f_plt[-1].elements:
            e.visible = False
        mlab.savefig(os.path.join(DIR, 'surf_B0.png'), size=(2000,2000), magnification=1)

        SCUTS['B']['visible'] = False
        mlab.savefig(os.path.join(DIR, 'surf_A2.png'), size=(2000,2000), magnification=1)

        for e in f_plt[-2].elements:
            e.visible = False
        mlab.savefig(os.path.join(DIR, 'surf_A1.png'), size=(2000,2000), magnification=1)

        for e in f_plt[-3].elements:
            e.visible = False
        mlab.savefig(os.path.join(DIR, 'surf_A0.png'), size=(2000,2000), magnification=1)

        SCUTS['C']['visible'] = True
        SCUTS['D']['visible'] = True
        SCUTS['B']['visible'] = True
        mlab.savefig(os.path.join(DIR, 'surf.png'), size=(2000,2000), magnification=1)
