import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as la
import numpy as np
import os, sys

from util.plot import SurfacePlot, SurfaceCut
from util.data import *

import dionysus as dio
from mayavi import mlab
import networkx as nx

SEED = 4869361 # np.random.randint(10000000) #
print('seed: %d' % SEED)
np.random.seed(SEED)

SURF_ARGS = {   'A' : {'min' : CUTS[0], 'max' : CUTS[1],    'color' : COLOR['green'],   'opacity' : 0.5,    'backface_culling' : True},
                'B' : {'min' : CUTS[1], 'max' : CUTS[2],    'color' : COLOR['blue'],    'opacity' : 0.5},
                'C' : {'min' : CUTS[2], 'max' : CUTS[3],    'color' : COLOR['purple'],  'opacity' : 0.5},
                'D' : {'min' : CUTS[3], 'max' : CUTS[4],    'color' : COLOR['yellow'],  'opacity' : 0.5}}

CONT_ARGS = {   'A_c' : {'scalar' : [CUTS[1]], 'color' : COLOR['green']},
                'B_c' : {'scalar' : [CUTS[2]], 'color' : COLOR['blue']},
                'C_c' : {'scalar' : [CUTS[3]], 'color' : COLOR['purple']}}

VIEW = {'default' : 'top',
        'side' : {  'view' : (-52., 88., 9.,
                        np.array([-1.00787402,  1.01587307,  0.6490444])),
                    'zoom' : 1.6, 'roll' : -89},
        'top' : {   'view' : (0.0, 0.0, 8.291977298839994,
                        np.array([-1.00648859,  1.05123171,  0.67399999])),
                    'zoom' : 1.6, 'roll' : -80}}

def grid_coord(v, N):
        return [v//N, v%N]

def set_view(view=None, zoom=None, roll=None):
    if view is not None:
        mlab.view(*view)
    if zoom is not None:
        gcf = mlab.gcf()
        gcf.scene.camera.parallel_scale = zoom
    if roll is not None:
        mlab.roll(roll)

def reset_view(key):
    set_view(**VIEW[key])

if __name__ == "__main__":

    N, WIDTH, HEIGHT = 32, 2, 1
    X_RNG = np.linspace(-WIDTH,WIDTH,WIDTH*N)
    Y_RNG = np.linspace(-HEIGHT,HEIGHT,HEIGHT*N)
    X, Y = np.meshgrid(X_RNG, Y_RNG)

    G = mk_gauss(X, Y, GAUSS_ARGS)


    SAVE = False
    AMT = 2
    cuti = 3
    cut = CUTS[cuti]
    subcut = CUTS[cuti-1]
    supcut = cut+0.2

    E = np.vstack((np.hstack((X[0], X[-1], X[:,0], X[:,-1])),
                    np.hstack((Y[0], Y[-1], Y[:,0], Y[:,-1])))).T

    _mask = G > cut
    mask = np.logical_and(G > cut, G < supcut)

    _M = nx.Graph()
    M = nx.Graph()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]:
                M.add_node((i,j))
                if i > 0 and mask[i-1,j]:
                    M.add_edge((i-1,j),(i,j))
                if j > 0 and mask[i,j-1]:
                    M.add_edge((i,j-1),(i,j))
            if _mask[i,j]:
                _M.add_node((i,j))
                if i > 0 and _mask[i-1,j]:
                    _M.add_edge((i-1,j),(i,j))
                if j > 0 and _mask[i,j-1]:
                    _M.add_edge((i,j-1),(i,j))

    cmp = nx.connected_components(M)
    Ms, Ps = [], []
    for comp in cmp:
        msk = np.zeros(mask.shape, bool)
        for i,j in comp:
            msk[i,j] = True
        Ms.append(msk)
        Ps.append(np.vstack((X[msk], Y[msk])).T)

    _cmp = nx.connected_components(_M)
    _Ms, Cs, Rs = [], [], []
    for comp in _cmp:
        msk = np.zeros(_mask.shape, bool)
        for i,j in comp:
            msk[i,j] = True
        _Ms.append(msk)
        P = np.vstack((X[msk], Y[msk])).T
        c = P.sum(axis=0) / len(P)
        Cs.append(c)
        Rs.append(min(la.norm(e - c) for e in E))

    Xs = [X.copy() for _ in Cs]
    Ys = [Y.copy() for _ in Cs]
    Gs = [G.copy() for _ in Cs]
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            for k, (c,r,P) in enumerate(zip(Cs, Rs, Ps)):
                if G[i,j] <= cut:
                    p = np.array([X[i,j], Y[i,j]])
                    d = la.norm(p - c)
                    v = (p - c) / d
                    if d < r: # and not any(Ms[l][i,j] for l in range(len(Ms)) if l != k):
                        b = r - d
                        q = min(P, key=lambda q: la.norm(p - q))
                        a = la.norm(q - c)
                        t = (G[i,j] / cut)
                        Xs[k][i,j], Ys[k][i,j] = c + v * a * (t + (1 - t) * b / (r - a))
                        Gs[k][i,j] = t * G[i,j] + (1 - t) * subcut
                        # if Gs[k][i,j] < subcut:
                        #     Gs[k][i,j] = subcut
                    else:
                        Xs[k][i,j], Ys[k][i,j] = c[0], c[1]
                        Gs[k][i,j] = subcut
                elif any(_Ms[l][i,j] for l in range(len(_Ms)) if l != k):
                    Xs[k][i,j], Ys[k][i,j] = c[0], c[1]
                    Gs[k][i,j] = subcut

    surfs = [mlab.mesh(x, y, g) for x,y,g in zip(Xs, Ys, Gs)]

    SCUTS = []
    for surf in surfs:
        surf.enable_contours = True
        _SURF_ARGS = {k : v.copy() for k,v in SURF_ARGS.items() if (v['min'] >= surf.contour.minimum_contour
                                                        and v['min'] <= surf.contour.maximum_contour)}
        _SURF_ARGS[max(_SURF_ARGS.keys())]['max'] = surf.contour.maximum_contour
        _SURF_ARGS[min(_SURF_ARGS.keys())]['min'] = surf.contour.minimum_contour
        surf.enable_contours = False

        SCUTS.append({k : SurfaceCut(surf.parent, k, **v) for k,v in _SURF_ARGS.items()})
        surf.visible = False

    gcf = mlab.gcf()
    scene = gcf.scene
    scene.parallel_projection = True
    scene.background = (1,1,1)
    reset_view('side')

    DIR = os.path.join('figures', 'alexander%d' % cuti)
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    if SAVE:
        mlab.savefig(os.path.join(DIR, 'surf.png'), size=(5000,5000))

    NOISE_AMT = 0.02
    THRESH = np.sqrt(2 * (2 / N) ** 2) / 2 + NOISE_AMT

    _I = [[i,j] for i in range(G.shape[0]) for j in range(G.shape[1]) if np.random.rand() / AMT < G[i,j] ** 2]
    _P = np.array([[X[i,j], Y[i,j], G[i,j]] for i,j in _I])
    NOISE = np.hstack(((2*np.random.rand(len(_P), 2) - 1) * NOISE_AMT/2, np.zeros((len(_P),1))))
    _PN = _P + NOISE
    R = dio.fill_rips(_PN[:,:2], 2, 2*THRESH)

    for k, (surf, x, y, g) in enumerate(zip(surfs, Xs, Ys, Gs)):
        # surf.visible = False
        # P = np.array([[x[i,j], y[i,j], g[i,j]] for i in range(g.shape[0]) for j in range(g.shape[1]) if (np.random.rand() < G[i,j] ** 2 and G[i,j] > subcut
        #                                                                                                 and not any(_Ms[l][i,j] for l in range(len(_Ms)) if l != k))]) # *G.max()
        I = [n for n, (i,j) in enumerate(_I) if (g[i,j] != subcut and not any(_Ms[l][i,j] for l in range(len(_Ms)) if l != k))]
        imap = {n : m for m,n in enumerate(I)}
        PN = np.array([[x[_I[l][0],_I[l][1]], y[_I[l][0],_I[l][1]], g[_I[l][0],_I[l][1]]] + NOISE[l] for l in I])
        # Z = (1.1+np.zeros(len(P))) * G.max()
        # NOISE = np.hstack(((2*np.random.rand(len(P), 2) - 1) * NOISE_AMT/2, np.zeros((len(P),1))))
        # PN = P + NOISE
        pt = mlab.points3d(PN[:,0], PN[:,1], PN[:,2], color=(0,0,0), scale_factor=0.02)
        pt.actor.property.lighting = False
        pt.glyph.glyph_source.glyph_source.phi_resolution = 16
        pt.glyph.glyph_source.glyph_source.theta_resolution = 16

        b = mlab.points3d(PN[:,0], PN[:,1], PN[:,2], color=COLOR['red'], scale_factor=2*THRESH, opacity=0.2)
        b.actor.property.lighting = False
        b.actor.property.frontface_culling = True
        b.glyph.glyph_source.glyph_source.phi_resolution = 32
        b.glyph.glyph_source.glyph_source.theta_resolution = 32

        # surf.save(os.path.join(DIR, 'cover.png'), (5000,5000))
        if SAVE:
            mlab.savefig(os.path.join(DIR, 'cover.png'), size=(5000,5000))

        b.visible = False

        # R = dio.fill_rips(PN[:,:2], 2, 2*THRESH)
        T = [[imap[v] for v in s] for s in R if s.dimension()==2 and all(v in imap for v in s)]
        val = np.array([[g[v//G.shape[0], v%g.shape[1]] for v in t] for t in T])
        t = mlab.triangular_mesh(PN[:,0], PN[:,1], PN[:,2], T, opacity=0.5, scalars=PN[:,2], color=COLOR['red'])
        t.actor.property.lighting = False
        t.actor.mapper.scalar_visibility = False
        t.enable_contours = True
        TRI_ARGS = {k : v.copy() for k,v in SURF_ARGS.items() if (v['min'] >= t.contour.minimum_contour
                                                                and v['min'] <= t.contour.maximum_contour)}
        TRI_ARGS[max(TRI_ARGS.keys())]['max'] = t.contour.maximum_contour
        TRI_ARGS[min(TRI_ARGS.keys())]['min'] = t.contour.minimum_contour
        t.enable_contours = False
        t.actor.mapper.scalar_visibility = False

        if SAVE:
            mlab.savefig(os.path.join(DIR, 'complex1.png'), size=(5000,5000))
        # surf.save(os.path.join(DIR, 'complex1.png'), (5000,5000))

        tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
        t.visible = False

        # surf.save(os.path.join(DIR, 'scalar1.png'), (5000,5000))
        if SAVE:
            mlab.savefig(os.path.join(DIR, 'scalar1.png'), size=(5000,5000))

        # for k,v in tri_cuts.items():
        #     v['visible'] = False

        # R = dio.fill_rips(PN[:,:2], 2, 4*THRESH)
        # T = [list(s) for s in R if s.dimension()==2]
        # val = np.array([[g[v//G.shape[0], v%G.shape[1]] for v in t] for t in T])
        # t = mlab.triangular_mesh(PN[:,0], PN[:,1], PN[:,2], T, opacity=0.5, scalars=PN[:,2], color=COLOR['red'])
        # t.actor.property.lighting = False
        # t.actor.mapper.scalar_visibility = False
        # t.enable_contours = True
        # TRI_ARGS = {k : v for k,v in SURF_ARGS.items() if v['max'] < t.contour.maximum_contour and v['min'] > t.contour.minimum_contour}
        # t.enable_contours = False
        # t.actor.mapper.scalar_visibility = False
        #
        # # surf.save(os.path.join(DIR, 'complex2.png'), (5000,5000))
        #
        # tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
        # t.visible = False
        #
        # # surf.save(os.path.join(DIR, 'scalar2.png'), (5000,5000))
