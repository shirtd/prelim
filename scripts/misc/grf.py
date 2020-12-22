import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as la
import numpy as np
import os, sys
from tqdm import tqdm

# from util.plot import SurfacePlot
from util.data import grf, gaussian
from mayavi import mlab
from util.persist import *
from scipy import ndimage

plt.ion()
fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,4))
plt.tight_layout()

SEED = 3042755 # np.random.randint(10000000) #
print('seed: %d' % SEED)
np.random.seed(SEED)

def remove_inf(d):
    return dio.Diagram([(p.birth, p.death) for p in d if p.death < np.inf])

def remove_infs(dgms):
    return [remove_inf(d) if len(d) else dio.Diagram([]) for d in dgms]

def get_dgms(G):
    filt = dio.fill_freudenthal(G)
    # hom = dio.homology_persistence(filt)
    # dgms = dio.init_diagrams(hom, filt)
    vmap = {v : filt[filt.index(dio.Simplex([v],0))].data for v in range(G.size)}
    res, rel = {'B' : [], 'C' : [], 'D' : []}, {'B' : [], 'C' : [], 'D' : []}
    for s in filt:
        for k in ['B', 'C', 'D']:
            if all(vmap[v] < CUT_MAP[k][0] for v in s):
                rel[k].append(s)
            elif all(vmap[v] >= CUT_MAP[k][0] for v in s):
                res[k].append(s)
    dgms_res, dgms_rel = {}, {}
    for k in ['B', 'C', 'D']:
        res_filt = dio.Filtration(res[k])
        rel_filt = dio.Filtration(rel[k])
        hom_res = dio.homology_persistence(res_filt)
        hom_rel = dio.homology_persistence(filt, relative=rel_filt)
        dgms_res[k] = remove_infs(dio.init_diagrams(hom_res, res_filt))
        dgms_rel[k] = remove_infs(dio.init_diagrams(hom_rel, filt))
    # return dgms, dgms_res, dgms_rel
    return None, dgms_res, dgms_rel

def down_sample(G, l):
    N, _ = G.shape
    _N, nrem = divmod(N, l)
    if nrem > 0:
        G = G[nrem//2:-nrem//2, nrem//2:-nrem//2]
    D = np.zeros((_N, _N), dtype=float)
    for j in range(_N):
        for i in range(_N):
            D[i, j] = G[i*l:(i+1)*l, j*l:(j+1)*l].sum() / (l ** 2)
    return D

# def block_mean(ar, fact):
#     assert isinstance(fact, int), type(fact)
#     sx, sy = ar.shape
#     X, Y = np.ogrid[0:sx, 0:sy]
#     regions = sy/fact * (X/fact) + Y/fact
#     res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
#     res.shape = (sx//fact, sy//fact)
#     return res

if __name__ == "__main__":
    EXP = -3
    N, WIDTH, HEIGHT = 256, 1, 1
    X_RNG = np.linspace(-WIDTH,WIDTH,WIDTH*N)
    Y_RNG = np.linspace(-HEIGHT,HEIGHT,HEIGHT*N)
    X, Y = np.meshgrid(X_RNG, Y_RNG)

    # G = (1 + grf(-4, N*N))% + (gaussian(X, Y, [0.3,0], [0.2, 0.2]) + 0.7*gaussian(X, Y, [-0.3,0], [0.2, 0.2]))


    _G = 2 * grf(EXP, N*N) - 0.5
    filt = dio.fill_freudenthal(_G)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)
    # dgms = get_res_dgm_sub(_G, 'A')

    # mlab.surf(X.T, Y.T, G)

    # dgms = get_res_dgm_sub(G, 'A')

    D_res, D_rel = {'B' : [[],[],[]], 'C' : [[],[],[]], 'D' : [[],[],[]]}, {'B' : [[],[],[]], 'C' : [[],[],[]], 'D' : [[],[],[]]}

    dgms_cut = {k : dio.Diagram([(p.birth, p.death) for p in dgms[1] if p.birth >= CUT_MAP[k][0]]) for k in ['B', 'C', 'D']}

    # for n in tqdm(range(32, 257)):
    for n in tqdm(range(32, -2, -2)):
        # np.random.seed(SEED)
        # G = 2 * grf(EXP, n*n)  - 0.5
        # n *= 4
        G = down_sample(_G, n if n > 0 else 1)
        _, dgms_res, dgms_rel = get_dgms(G)
        for k in ['B','C','D']:
            D_res[k][1].append(dio.bottleneck_distance(dgms_cut[k], dgms_res[k][1]))
            D_rel[k][1].append(dio.bottleneck_distance(dgms_cut[k], dgms_rel[k][1]))
            # for i, (a,b) in enumerate(zip(dgms, dgms_res[k])):
            #     a = dio.Diagram([(p.birth, p.death) for p in a if p.birth >= CUT_MAP[k][0]])
            #     D_res[k][i].append(dio.bottleneck_distance(a,b))
            # for i, (a,b) in enumerate(zip(dgms, dgms_rel[k])):
            #     a = dio.Diagram([(p.birth, p.death) for p in a if p.birth >= CUT_MAP[k][0]])
            #     D_rel[k][i].append(dio.bottleneck_distance(a,b))

    for j, k in enumerate(['B','C','D']):
        ax[j].plot(D_res[k][1], alpha=0.5)
        ax[j].plot(D_rel[k][1], alpha=0.5)#$, linestyle='dotted')
        # for i,d in enumerate(D_res[k][1:-1]):
        #     ax[i,j].plot(d, alpha=0.5)
        # for i,d in enumerate(D_rel[k][:-1]):
        #     ax[i,j].plot(d, alpha=0.5)#$, linestyle='dotted')
    #
    #
    #
    #
    # # s = mlab.surf(X.T, Y.T, G)
    # #
    # # TYPE = 'sub' # 'super' #
    # # FILT = 'res' # 'rel' #
    # #
    # # if TYPE == 'sub':
    # #     if FILT == 'res':
    # #         dgms = {name : get_res_dgm_sub(G, name) for name in CUT_MAP.keys()}
    # #     elif FILT == 'rel':
    # #         dgms = {name : get_rel_dgm_sub(G, name) for name in CUT_MAP.keys()}
    # # elif TYPE == 'super':
    # #     if FILT == 'res':
    # #         dgms = {name : get_res_dgm_super(G, name) for name in CUT_MAP.keys()}
    # #     elif FILT == 'rel':
    # #         dgms = {name : get_rel_dgm_super(G, name) for name in CUT_MAP.keys()}
    # #
    # # plot_barcodes(ax[0], dgms['A'], TYPE, 1)
    # # plot_barcodes(ax[1], dgms['B'], TYPE, 1)
    # # plot_barcodes(ax[2], dgms['C'], TYPE, 1)
    # # plot_barcodes(ax[3], dgms['D'], TYPE, 1)
