import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as la
import numpy as np
import os, sys

from util.plot import SurfacePlot, SurfaceCut
from util.data import *

import dionysus as dio
from mayavi import mlab

SEED = 6611635 # np.random.randint(10000000) # 4869361 # 1743801 # 8359430 # 5880992 #
print('seed: %d' % SEED)
np.random.seed(SEED)

SURF_ARGS = {   'A' : {'min' : CUTS[0], 'max' : CUTS[1],    'color' : COLOR['green'],   'opacity' : 0.5},#,    'backface_culling' : True},
                'B' : {'min' : CUTS[1], 'max' : CUTS[2],    'color' : COLOR['blue'],    'opacity' : 0.5},
                'C' : {'min' : CUTS[2], 'max' : CUTS[3],    'color' : COLOR['purple'],  'opacity' : 0.5},
                'D' : {'min' : CUTS[3], 'max' : CUTS[4],    'color' : COLOR['yellow'],  'opacity' : 0.5}}

OMEGA = CUTS[2]
CONT_ARGS = {   'A_c' : {'scalar' : [CUTS[0]], 'color' : COLOR['black']},
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
    # DIR = os.path.join('figures', 'cover')
    DIR = os.path.join('figures', 'rips2')
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    SAVE = True

    MOD = 4
    AMT = 1
    OFF = 1.0 * 2
    N, WIDTH, HEIGHT = 32, 2, 1
    NOISE_AMT = 0.2
    THRESH = OFF * (np.sqrt(2 * (2 / N) ** 2) / 2 + NOISE_AMT)

    # N, WIDTH, HEIGHT = 512, 2, 1
    X_RNG = np.linspace(-WIDTH,WIDTH,WIDTH*N)
    Y_RNG = np.linspace(-HEIGHT,HEIGHT,HEIGHT*N)
    X, Y = np.meshgrid(X_RNG, Y_RNG)

    G = mk_gauss(X, Y, GAUSS_ARGS)
    surf = SurfacePlot(X, Y, G, SURF_ARGS, CONT_ARGS, VIEW)
    surf.reset_view('top')
    # surf['cont']['A_c']['visible'] = True
    # surf['cont']['A_c']._o.actor.property.line_width = 10

    # P = np.array([[Y[i,j]-1, X[i,j]+1, G[i,j]] for i in range(G.shape[0]) for j in range(G.shape[1]) if G[i,j] > 0.03 and np.random.rand() > 1/AMT]) # *G.max()
    # P = np.array([[Y[i,j]-1, X[i,j]+1, G[i,j]] for i in range(G.shape[0]) for j in range(G.shape[1]) if np.random.rand() < 1/AMT and G[i,j] >= CUTS[0] and i % MOD == 1 and j % MOD == 3]) # *G.max()
    P = np.array([[Y[i,j]-1, X[i,j]+1, G[i,j]] for i in range(G.shape[0]) for j in range(G.shape[1]) if i % MOD == 2 and j % MOD == 2]) # *G.max()

    NOISE = np.hstack(((2*np.random.rand(len(P), 2) - 1) * NOISE_AMT/2, np.zeros((len(P),1))))
    # PN = P + NOISE
    PN = []
    for (x,y,z), (nx,ny,nz) in zip(P, NOISE):
        i, j = int(HEIGHT*N * (x + 2) / 2), int(WIDTH*N * (y + 1) / 4)
        ni, nj = int(HEIGHT*N * (x + nx + 2) / 2), int(WIDTH*N * (y + ny + 1) / 4)
        if 0 <= ni < G.shape[0] and 0 <= nj < G.shape[1] and G[ni,nj] > CUTS[0]:
            PN.append([x+nx,y+nx,z])
        elif 0 <= ni < G.shape[0] and 0 <= j < G.shape[1] and G[ni,j] > CUTS[0]:
            PN.append([x+nx,y,z])
        elif 0 <= i < G.shape[0] and 0 <= nj < G.shape[1] and G[i,nj] > CUTS[0]:
            PN.append([x,y+ny,z])
        elif z > CUTS[0]:
            PN.append([x,y,z])
        # elif 0 <= i < G.shape[0] and 0 <= j < G.shape[1] and G[i,j] > CUTS[0]:
        #     PN.append([x,y,z])

    # _PN = []
    # for x,y,z in PN:
    #     i, j = int(HEIGHT*N * (x + 2) / 2),int(WIDTH*N * (y + 1) / 4)
    #     if 0 <= i < G.shape[0] and 0 <= j < G.shape[1] and G[i,j] > CUTS[0]:
    #         _PN.append([x,y,z])

    PN = np.array(PN)
    Z = (1.1+np.zeros(len(PN))) * G.max()


    p = mlab.points3d(PN[:,0], PN[:,1], Z, color=(0,0,0), scale_factor=0.025)
    p.actor.property.lighting = False
    p.glyph.glyph_source.glyph_source.phi_resolution = 16
    p.glyph.glyph_source.glyph_source.theta_resolution = 16

    D = np.vstack((Y.flatten()-1, X.flatten()+1)).T
    I = [i for i,v in enumerate(G.flatten()) if 0.0 < v <= 0.06]
    B = D[I]

    Qidx = []
    for l,(x,y,z) in enumerate(PN):
        i, j = int(HEIGHT*N * (x + 2) / 2), int(WIDTH*N * (y + 1) / 4)
        if 0 <= ni < G.shape[0] and 0 <= nj < G.shape[1] and G[i,j] < OMEGA - THRESH/3:
            Qidx.append(l)
    # Qidx = [i for i,p in enumerate(PN) if p[2] <= OMEGA] #min(la.norm(p[:2] - q) for q in B) <= 2*THRESH]# or p[2] <= 0.05]
    Nidx = list(set(range(len(PN))) - set(Qidx))

    pq = mlab.points3d(PN[Qidx,0], PN[Qidx,1], Z[Qidx], color=(0,0,0), scale_factor=0.025)
    pq.actor.property.lighting = False
    pq.glyph.glyph_source.glyph_source.phi_resolution = 16
    pq.glyph.glyph_source.glyph_source.theta_resolution = 16
    pq.visible = False

    b = mlab.points3d(PN[:,0], PN[:,1], Z[:], color=COLOR['red'], scale_factor=2*THRESH, opacity=0.2)
    b.actor.property.lighting = False
    b.actor.property.frontface_culling = True
    b.glyph.glyph_source.glyph_source.phi_resolution = 32
    b.glyph.glyph_source.glyph_source.theta_resolution = 32

    b.visible = False
    # c.visible = False

    bp = mlab.points3d(PN[Nidx,0], PN[Nidx,1], Z[Nidx], color=COLOR['red'], scale_factor=2*THRESH, opacity=0.2)
    bp.actor.property.lighting = False
    bp.actor.property.frontface_culling = True
    bp.glyph.glyph_source.glyph_source.phi_resolution = 32
    bp.glyph.glyph_source.glyph_source.theta_resolution = 32

    bp.visible = False

    bq = mlab.points3d(PN[Qidx,0], PN[Qidx,1], Z[Qidx], color=COLOR['green'], scale_factor=2*THRESH, opacity=0.2)
    bq.actor.property.lighting = False
    bq.actor.property.frontface_culling = True
    bq.glyph.glyph_source.glyph_source.phi_resolution = 32
    bq.glyph.glyph_source.glyph_source.theta_resolution = 32

    bq.visible = False


    R = dio.fill_rips(PN[:,:2], 2, 2*THRESH)
    T = [list(s) for s in R if s.dimension()==2]
    # val = np.array([[G[v//G.shape[0], v%G.shape[1]] for v in t] for t in T])
    t = mlab.triangular_mesh(PN[:,0], PN[:,1], Z, T, opacity=0.5, scalars=PN[:,2], color=COLOR['red'])
    t.actor.property.lighting = False
    t.actor.mapper.scalar_visibility = False
    t.enable_contours = True
    TRI_ARGS = {k : v for k,v in SURF_ARGS.items() if v['max'] <= t.contour.maximum_contour and v['min'] >= t.contour.minimum_contour}
    TRI_ARGS['A'] = SURF_ARGS['A'].copy()
    TRI_ARGS['D'] = SURF_ARGS['D'].copy()
    TRI_ARGS['A']['min'] = t.contour.minimum_contour
    TRI_ARGS['D']['max'] = t.contour.maximum_contour
    t.enable_contours = False
    t.actor.mapper.scalar_visibility = False

    tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
    # for k,v in tri_cuts.items():
    #     v._o.actor.property.edge_visibility = True
    #     v._o.actor.property.line_width = 0.1
    # t.
    t.actor.property.representation = 'wireframe'
    t.actor.property.color = (0,0,0)
    t.visible = False

    # t.parent.visible = False

    E = [list(s) for s in R if s.dimension()==1]
    es = []
    for uv in E:
        e = mlab.plot3d(PN[uv,0], PN[uv,1], Z[uv], color=COLOR['black'])#, line_width=0.01)
        e.parent.parent.filter.radius = 0.002
        e.actor.property.lighting = False
        e.actor.mapper.scalar_visibility = False
        e.parent.parent
        es.append(e)



    # t.visible = False
    # t.actor.property.lighting = False
    # t.actor.mapper.scalar_visibility = False
    # t.enable_contours = True
    # TRI_ARGS = {'P' : {'min' : 0, 'max' : 0.1,    'color' : COLOR['red'],   'opacity' : 0.2},
    #             'Q' : {'min' : 0.1, 'max' : 1,    'color' : COLOR['blue'],   'opacity' : 0.2}}
    # t.enable_contours = False
    # t.actor.mapper.scalar_visibility = False
    # tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
    # t.visible = False
    #
    # tri_cuts['Q']._o.contour.auto_update_range = False
    # tri_cuts['Q']._o.contour._data_max = 2.0
    # tri_cuts['Q']._o.contour.maximum_contour = 2.0

    # SurfaceCut(t.parent, 'A', min=t.contour.minimum_contour, max=0.3, color=SURF_ARGS['A']['color'], opacity=0.5)

    # surf.save(os.path.join(DIR, 'dump.png'), (10,10))
    # surf.save(os.path.join(DIR, 'dump.png'), (10,10))
    # surf.save(os.path.join(DIR, 'complex.png'), (3000,3000))

    RQ = dio.fill_rips(PN[Qidx,:2], 2, 2*THRESH)
    TQ = [list(s) for s in RQ if s.dimension()==2]
    # val = np.array([[G[v//G.shape[0], v%G.shape[1]] for v in t] for t in T])
    tq = mlab.triangular_mesh(PN[Qidx,0], PN[Qidx,1], Z[Qidx], TQ, opacity=0.5, scalars=PN[Qidx,2], color=COLOR['green'])
    tq.actor.property.lighting = False
    tq.actor.mapper.scalar_visibility = False
    # tq.enable_contours = True
    tq.visible = False

    EQ = [list(s) for s in RQ if s.dimension()==1]
    eqs = []
    for i,j in EQ:
        uv = [Qidx[i], Qidx[j]]
        e = mlab.plot3d(PN[uv,0], PN[uv,1], Z[uv], color=COLOR['black'])#, line_width=0.01)
        e.parent.parent.filter.radius = 0.002
        e.actor.property.lighting = False
        e.actor.mapper.scalar_visibility = False
        e.parent.parent
        eqs.append(e)
        e.visible = False




    # TRI_ARGS = {k : v for k,v in SURF_ARGS.items() if v['max'] <= t.contour.maximum_contour and v['min'] >= t.contour.minimum_contour}
    # TRI_ARGS['A'] = SURF_ARGS['A'].copy()
    # TRI_ARGS['D'] = SURF_ARGS['D'].copy()
    # TRI_ARGS['A']['min'] = t.contour.minimum_contour
    # TRI_ARGS['D']['max'] = t.contour.maximum_contour
    # t.enable_contours = False
    # t.actor.mapper.scalar_visibility = False
    #
    # tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
    # # for k,v in tri_cuts.items():
    # #     v._o.actor.property.edge_visibility = True
    # #     v._o.actor.property.line_width = 0.1
    # # t.
    # t.actor.property.representation = 'wireframe'
    # t.actor.property.color = (0,0,0)

    if SAVE:
        surf.save(os.path.join(DIR, 'dump.png'), (10,10))
        surf.save(os.path.join(DIR, 'dump.png'), (10,10))


        surf.save(os.path.join(DIR, 'complex.png'), (3000,3000))
        t.parent.visible = False

        surf.save(os.path.join(DIR, 'skeleton.png'), (3000,3000))
        for e in es:
            e.visible = False

        for e in eqs:
            e.visible = True

        pq.visible = True
        p.visible = False

        surf.save(os.path.join(DIR, 'Qskeleton.png'), (3000,3000))

        for e in eqs:
            e.visible = False

        # t.parent.visible = False
        # t.visible = True
        # surf.save(os.path.join(DIR, 'skeleton.png'), (3000,3000))
        # t.visible = False

        pq.visible = False
        p.visible = True

        b.visible = True
        # c.visible = True
        surf.save(os.path.join(DIR, 'cover.png'), (3000,3000))
        b.visible = False

        bq.visible = True
        pq.visible = True
        p.visible = False
        surf.save(os.path.join(DIR, 'Qcover.png'), (3000,3000))
        bp.visible = True
        pq.visible = False
        p.visible = True
        surf.save(os.path.join(DIR, 'PQcover.png'), (3000,3000))
        bq.visible = False
        bp.visible = False
        # c.visible = False
        surf.save(os.path.join(DIR, 'samples.png'), (3000,3000))

        p.visible = False
        surf.save(os.path.join(DIR, 'surf.png'), (3000,3000))


        surf.ctl.visible = False
        p.visible = True
        t.parent.visible = True
        t.visible=False
        for e in es:
            e.visible = True
        surf.save(os.path.join(DIR, 'complex_nosurf.png'), (3000,3000))
    #
    # if SAVE:
    #     surf.save(os.path.join(DIR, 'complex1.png'), (3000,3000))
    #
    # tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
    # t.visible = False
    #
    # if SAVE:
    #     surf.save(os.path.join(DIR, 'scalar1.png'), (3000,3000))
    #
    # # for k,v in tri_cuts.items():
    # #     v['visible'] = False
    # #
    # # R = dio.fill_rips(PN[:,:2], 2, 4*THRESH)
    # # T = [list(s) for s in R if s.dimension()==2]
    # # val = np.array([[G[v//G.shape[0], v%G.shape[1]] for v in t] for t in T])
    # # t = mlab.triangular_mesh(PN[:,0], PN[:,1], Z, T, opacity=0.5, scalars=PN[:,2], color=COLOR['red'])
    # # t.actor.property.lighting = False
    # # t.actor.mapper.scalar_visibility = False
    # # t.enable_contours = True
    # # TRI_ARGS = {k : v for k,v in SURF_ARGS.items() if v['max'] < t.contour.maximum_contour and v['min'] > t.contour.minimum_contour}
    # # t.enable_contours = False
    # # t.actor.mapper.scalar_visibility = False
    # #
    # # if SAVE:
    # #     surf.save(os.path.join(DIR, 'complex2.png'), (3000,3000))
    # #
    # # tri_cuts = {k : SurfaceCut(t.parent, k, **v) for k,v in TRI_ARGS.items()}
    # # t.visible = False
    # #
    # # if SAVE:
    # #     surf.save(os.path.join(DIR, 'scalar2.png'), (3000,3000))
    # #
    # # # s = mlab.surf(X.T, Y.T, G)
