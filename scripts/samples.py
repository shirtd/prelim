import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as la
import numpy as np
import os, sys

from util.plot import SurfacePlot, SurfaceCut
from util.data import *

import dionysus as dio
from mayavi import mlab

SEED = 4869361 # np.random.randint(10000000) #
print('seed: %d' % SEED)
np.random.seed(SEED)

SURF_ARGS = {   'A' : {'min' : CUTS[0], 'max' : CUTS[1],    'color' : COLOR['green'],   'opacity' : 0.1},#,    'backface_culling' : True},
                'B' : {'min' : CUTS[1], 'max' : CUTS[2],    'color' : COLOR['blue'],    'opacity' : 0.5},
                'C' : {'min' : CUTS[2], 'max' : CUTS[3],    'color' : COLOR['purple'],  'opacity' : 0.5},
                'D' : {'min' : CUTS[3], 'max' : CUTS[4],    'color' : COLOR['yellow'],  'opacity' : 0.5}}

OMEGA = CUTS[2]
CONT_ARGS = {   'omega' : {'scalar' : [OMEGA], 'color' : COLOR['black']},
                'A_c' : {'scalar' : [CUTS[1]], 'color' : COLOR['green']},
                'B_c' : {'scalar' : [CUTS[2]], 'color' : COLOR['blue']},
                'C_c' : {'scalar' : [CUTS[3]], 'color' : COLOR['purple']}}

VIEW = {'default' : 'top',
        'side' : {  'view' : (-11.800830502323867, 80.88795919149756, 9.035877007511106,
                        np.array([-1.00787402,  1.01587307,  0.6490444])),
                    'zoom' : 1.6, 'roll' : -89},
        'top' : {   'view' : (0.0, 0.0, 8.291977298839994,
                        np.array([-1.00648859,  1.05123171,  0.67399999])),
                    'zoom' : 1.6, 'roll' : -80}}

def grid_coord(v, N):
        return [v//N, v%N]



if __name__ == "__main__":
    DIR = os.path.join('figures', 'samples_1')
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    SAVE = True

    AMT = 3
    OFF = 1
    N, WIDTH, HEIGHT = 32, 2, 1
    NOISE_AMT = 0.05
    THRESH = OFF * (np.sqrt(2 * (2 / N) ** 2) / 2 + NOISE_AMT)

    # N, WIDTH, HEIGHT = 512, 2, 1
    X_RNG = np.linspace(-WIDTH,WIDTH,WIDTH*N)
    Y_RNG = np.linspace(-HEIGHT,HEIGHT,HEIGHT*N)
    X, Y = np.meshgrid(X_RNG, Y_RNG)

    G = mk_gauss(X, Y, GAUSS_ARGS)
    surf = SurfacePlot(X, Y, G, SURF_ARGS, CONT_ARGS, VIEW)
    surf['cont']['omega']['visible'] = True
    surf['cont']['omega']._o.actor.property.line_width = 10
    surf.reset_view('top')

    # P = np.array([[Y[i,j]-1, X[i,j]+1, G[i,j]] for i in range(G.shape[0]) for j in range(G.shape[1]) if np.random.rand()/AMT < G[i,j] ** 2]) # *G.max()
    P = np.array([[Y[i,j]-1, X[i,j]+1, G[i,j]] for i in range(G.shape[0]) for j in range(G.shape[1]) if np.random.rand()/AMT < G[i,j] ** 2]) # *G.max()
    Z = (1.1+np.zeros(len(P))) * G.max()
    NOISE = np.hstack(((2*np.random.rand(len(P), 2) - 1) * NOISE_AMT/2, np.zeros((len(P),1))))
    PN = P + NOISE
    p = mlab.points3d(PN[:,0], PN[:,1], Z, color=(0,0,0), scale_factor=0.02)
    p.actor.property.lighting = False
    p.glyph.glyph_source.glyph_source.phi_resolution = 16
    p.glyph.glyph_source.glyph_source.theta_resolution = 16

    b = mlab.points3d(PN[:,0], PN[:,1], Z, color=COLOR['red'], scale_factor=2*THRESH, opacity=0.2)
    b.actor.property.lighting = False
    b.actor.property.frontface_culling = True
    b.glyph.glyph_source.glyph_source.phi_resolution = 32
    b.glyph.glyph_source.glyph_source.theta_resolution = 32

    b.visible = False

    R = dio.fill_rips(PN[:,:2], 2, 2*THRESH)
    T = [list(s) for s in R if s.dimension()==2]
    val = np.array([[G[v//G.shape[0], v%G.shape[1]] for v in t] for t in T])
    t = mlab.triangular_mesh(PN[:,0], PN[:,1], Z, T, opacity=0.5, scalars=PN[:,2], color=COLOR['red'])
    t.actor.property.lighting = False
    t.actor.mapper.scalar_visibility = False
    t.enable_contours = True
    TRI_ARGS = {k : v for k,v in SURF_ARGS.items() if v['max'] < t.contour.maximum_contour and v['min'] > t.contour.minimum_contour}
    t.enable_contours = False
    t.actor.mapper.scalar_visibility = False

    tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
    # t.visible = False
    t.actor.property.representation = 'wireframe'
    t.actor.property.color = (0,0,0)

    if SAVE:
        surf.save(os.path.join(DIR, 'dump.png'), (10,10))
        surf.save(os.path.join(DIR, 'dump.png'), (10,10))

        surf.save(os.path.join(DIR, 'scalar1.png'), (3000,3000))
        # for r in tri_cuts.values():
        #     r['visible'] = False

        # t.visible = True
        # surf.save(os.path.join(DIR, 'complex1.png'), (3000,3000))

        t.parent.visible = False
        b.visible = True
        surf.save(os.path.join(DIR, 'cover.png'), (3000,3000))

        b.visible = False
        surf.save(os.path.join(DIR, 'samples.png'), (3000,3000))

        p.visible = False
        surf.save(os.path.join(DIR, 'surf.png'), (3000,3000))

    # for k,v in tri_cuts.items():
    #     v['visible'] = False
    #
    # R = dio.fill_rips(PN[:,:2], 2, 4*THRESH)
    # T = [list(s) for s in R if s.dimension()==2]
    # val = np.array([[G[v//G.shape[0], v%G.shape[1]] for v in t] for t in T])
    # t = mlab.triangular_mesh(PN[:,0], PN[:,1], Z, T, opacity=0.5, scalars=PN[:,2], color=COLOR['red'])
    # t.actor.property.lighting = False
    # t.actor.mapper.scalar_visibility = False
    # t.enable_contours = True
    # TRI_ARGS = {k : v for k,v in SURF_ARGS.items() if v['max'] < t.contour.maximum_contour and v['min'] > t.contour.minimum_contour}
    # t.enable_contours = False
    # t.actor.mapper.scalar_visibility = False
    #
    # if SAVE:
    #     surf.save(os.path.join(DIR, 'complex2.png'), (3000,3000))
    #
    # tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
    # t.visible = False
    #
    # if SAVE:
    #     surf.save(os.path.join(DIR, 'scalar2.png'), (3000,3000))
    #
    # # s = mlab.surf(X.T, Y.T, G)
