from multiprocessing import Pool
from functools import partial
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.random import rand
from itertools import combinations
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib import cm

def stuple(s, *args, **kw):
    return tuple(sorted(s, *args, **kw))

def pmap(fun, x, max_cores=None, *args, **kw):
    pool = Pool(max_cores)
    f = partial(fun, *args, **kw)
    try:
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def circumcenter(t):
    t = t.T
    f = np.array([( t[0,1] - t[0,0] ) ** 2 + ( t[1,1] - t[1,0] ) ** 2,
            ( t[0,2] - t[0,0] ) ** 2 + ( t[1,2] - t[1,0] ) ** 2])
    top = np.array([( t[1,2] - t[1,0] ) * f[0] - ( t[1,1] - t[1,0] ) * f[1],
                - ( t[0,2] - t[0,0] ) * f[0] + ( t[0,1] - t[0,0] ) * f[1]])
    det  = (t[1,2]-t[1,0] ) * (t[0,1]-t[0,0]) - (t[1,1]-t[1,0]) * (t[0,2]-t[0,0])
    if det == 0: return None
    return np.array([t[0,0] + 0.5 * top[0] / det, t[1,0] + 0.5 * top[1] / det])

def plot_dgm(axis, dgm, lim=None, omega=None, maxdim=None, clear=False, show=False):
    if clear:
        axis.cla()
    lim = dgm.lim if lim is None else lim
    if len(dgm.diagram):
        maxdim = max(dgm.diagram) if maxdim is None else maxdim
    else:
        return
    axis.plot([0, lim],[lim, lim], c='black', ls=':', alpha=0.5, zorder=0)
    axis.plot([lim, lim],[lim, 2*lim], c='black', ls=':', alpha=0.5, zorder=0)
    axis.plot([0, 2*lim], [0, 2*lim], c='black', zorder=0, alpha=0.5)
    if omega is not None:
        axis.plot([0, omega],[omega, omega], c='black', ls=':', alpha=0.5, zorder=0)
        axis.plot([omega, omega],[omega, 2*lim], c='black', ls=':', alpha=0.5, zorder=0)
    for dim in range(maxdim+1):
        d = dgm.as_np(dim, lim)
        axis.scatter(d[:,0], d[:,1], alpha=0.5, zorder=1, s=5,
                        label='dim = %d (%d)' % (dim, len(dgm.get_inf(dim))))
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], loc=4)
    # axis.legend(loc=4)
    if show:
        plt.pause(0.1)

def plot_dgm_dio(axis, dgm, lim=None, maxdim=None, clear=False, show=False):
    if clear:
        axis.cla()
    lim = dgm.lim if lim is None else lim
    axis.plot([0, lim],[lim, lim], c='black', ls=':', alpha=0.5, zorder=0)
    axis.plot([lim, lim],[lim, 2*lim], c='black', ls=':', alpha=0.5, zorder=0)
    axis.plot([0, 2*lim], [0, 2*lim], c='black', zorder=0, alpha=0.5)
    for dim, dg in enumerate(dgm[:-1]):
        if len(dg):
            d = np.array([[p.birth, p.death if p.death < np.inf else 2*lim] for p in dg])
        else:
            d = np.ndarray((0,2), dtype=float)
        axis.scatter(d[:,0], d[:,1], alpha=0.5, zorder=1, s=5, label='dim = %d' % dim)
    axis.legend()
    if show:
        plt.pause(0.1)

def plot_1_chain(axis, dgm, pt, color='blue', zorder=5, show=False, torus=True):
    l = 2*np.pi
    tiles = [[0,0],[l,0],[-l,0],[0,l],[0,-l],
            [l,l],[-l,l],[-l,-l],[l,-l]]
    chain = dgm.get_chain(pt)
    if torus:
        E = [dgm.F.points[list(s)] + s.offset for s in chain]
        for e in E:
            for a,b in tiles:
                axis.plot(e[:,0]+a, e[:,1]+b, c=color, zorder=zorder)
    else:
        E = [dgm.F.points[list(s)] for s in chain]
        for e in E:
            axis.plot(e[:,0], e[:,1], c=color, zorder=zorder)
    if show:
        plt.pause(0.1)

def plot_2_boundary(axis, dgm, pt, color='red', zorder=5, show=False, torus=True):
    l = 2*np.pi
    tiles = [[0,0],[l,0],[-l,0],[0,l],[0,-l],
            [l,l],[-l,l],[-l,-l],[l,-l]]
    chain = dgm.get_chain(pt)
    bdy = dgm.F.chain_boundary(chain)
    if torus:
        E = [dgm.F.points[list(s)] + s.offset for s in bdy]
        for e in E:
            for a,b in tiles:
                axis.plot(e[:,0]+a, e[:,1]+b, c=color, zorder=zorder)
    else:
        E = [dgm.F.points[list(s)] for s in bdy]
        for e in E:
            axis.plot(e[:,0], e[:,1], c=color, zorder=zorder)
    if show:
        plt.pause(0.1)
