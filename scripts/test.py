import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from util.bottleneck import *
import numpy.linalg as la
import numpy as np
import os, sys

from util.plot import *
from util.data import *

from load_surf import VIEW, tup_d, edges_to_path

from util.diagram import *

class FeaturePlot:
    def __init__(self, chain, X, Y, G, chain_color=COLOR['red'], rad=0.007):
        C = [[grid_coord(v, len(X)) for v in s] for s in chain]
        paths = edges_to_path(C)
        chains = [np.array([[Y[a,b], X[a,b], G[a,b]] for a,b in p]) for p in paths]
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

if __name__ == "__main__":
    fname = sys.argv[1]
    self = load_barcodes(fname)
    G = self.field
    X_RNG = np.linspace(-1,1,self.N)
    Y_RNG = np.linspace(-1,1,self.N)
    X, Y = np.meshgrid(X_RNG, Y_RNG)

    CMAP = [(COLOR['green'],    (0, self.cuts[0])),
            (COLOR['blue'],     (self.cuts[0], self.cuts[1])),
            (COLOR['purple'],   (self.cuts[1], self.cuts[2])),
            (COLOR['yellow'],   (self.cuts[2], 1))]

    LABELS = ['A', 'B', 'C', 'D']#, 'E']

    SURF_ARGS = {l : {'min' : a, 'max' : b, 'color' : c, 'opacity' : 0.5} for l,(c,(a,b)) in zip(LABELS, CMAP)}
    CONT_ARGS = {'_'.join((l,'c')) : {'scalar' : [b], 'color' : c} for l,(c,(a,b)) in zip(LABELS, CMAP)}

    THRESH = 4 * np.sqrt(2 * (2 / self.N) ** 2)
    barcode = self.barcodes[-1]

    surf = SurfacePlot(X, Y, G, SURF_ARGS, CONT_ARGS, VIEW)
    surf.reset_view('top')

    di = 0
    name = os.path.splitext(os.path.basename(fname))[0]
    dir = os.path.join('figures', 'experiments', 'relative')#, name)
    while os.path.exists(os.path.join(dir, '%s-%d' % (name, di))):
        di += 1
    dir = os.path.join(dir, '%s-%d' % (name, di))
    print('creating directory %s' % dir)
    os.makedirs(dir)

    surf.save(os.path.join(dir, 'full-surf_top.png'), (2000, 2000))

    vw = view()
    view(vw[0], 80, vw[2], vw[3])
    surf.save(os.path.join(dir, 'full-surf_side.png'), (2000, 2000))
    surf.reset_view('top')

    plt.ion()

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 4))

    plot_diagram(ax[0], self.full_barcode[1], CMAP, THRESH, alpha=0.75, size=4, zorder=1)

    ax[0].set_title(r"$\mathrm{H}_1$ full")

    plt.tight_layout()

    G.dtype = float
    dio_filt = dio.fill_freudenthal(G)

    # for j, cut in enumerate(self.cuts):
    #
    #     surf['cut'][LABELS[j]]['opacity'] = 0.1
    #
    #     Fres = DioFilt(dio_filt, self.field, cut, False, True, False)
    #     dgm_res = phcol(Fres)
    #
    #     Frel = DioFilt(dio_filt, self.field, cut, True, False, False)
    #     dgm_rel = phcol(Frel)
    #
    #
    #     ax[1].cla()
    #     plot_diagram(ax[1], dgm_res.as_list(1), CMAP, THRESH, alpha=0.75, size=4, zorder=1)
    #
    #     ax[2].cla()
    #     plot_diagram(ax[2], dgm_rel.as_list(2), CMAP, THRESH, alpha=0.75, size=4, zorder=1)
    #
    #
    #     ax[1].set_title(r"$\mathrm{H}_1$ restricted (to $B_{%0.1f}$)" % cut)
    #     ax[2].set_title(r"$\mathrm{H}_2$ relative (to $B_{%0.1f}$)" % cut)
    #
    #     plt.savefig(os.path.join(dir, 'full-dgm-%d.png' % j), dpi=300)
    #
    #     f_plt, p_plt = [], []
    #
    #     i = 0
    #     for p,q in zip(reversed(dgm_res.get_inf(1)), dgm_rel.get_inf(2)):
    #         if q.birth - p.birth >= THRESH:
    #             p_plt.append(ax[0].scatter(p.birth, q.birth, s=15, marker='D', color=get_color([p.birth, q.birth], CMAP), zorder=3))
    #             p_plt.append(ax[0].text(p.birth-0.07, q.birth-0.07, r"$(%0.2f,%0.2f)$" % (p.birth, q.birth), fontsize=7))
    #
    #             p_plt.append(ax[1].scatter(p.birth, 1.2, s=15, marker='D', color=get_color([p.birth, p.death], CMAP), zorder=3))
    #             p_plt.append(ax[1].text(p.birth-0.07, 1.2-0.07, r"$(%0.2f,\infty)$" % p.birth, fontsize=7))
    #
    #             p_plt.append(ax[2].scatter(q.birth, 1.2, s=15, marker='D', color=get_color([q.birth, q.death], CMAP), zorder=3))
    #             p_plt.append(ax[2].text(q.birth-0.07, 1.2-0.07, r"$(%0.2f,\infty)$" % q.birth, fontsize=7))
    #
    #             csrel = dgm_rel.get_chain(q)
    #             dcrel = chain_boundary(csrel)
    #             f_plt.append(FeaturePlot(dcrel, X, Y, G, COLOR['red']))
    #
    #             csres = dgm_res.get_chain(p)
    #             f_plt.append(FeaturePlot(csres, X, Y, G, (0,0,0)))
    #
    #             surf.reset_view('top')
    #             surf.save(os.path.join(dir, 'surf_top-%d_%d.png' % (j,i)), (3000, 3000))
    #             vw = view()
    #             view(vw[0], 80, vw[2], vw[3])
    #             surf.save(os.path.join(dir, 'surf_side-%d_%d.png' % (j,i)), (3000, 3000))
    #             surf.reset_view('top')
    #
    #             plt.pause(0.1)
    #             plt.savefig(os.path.join(dir, 'dgm-%d_%d.png' % (j,i)), dpi=300)
    #
    #             while len(f_plt):
    #                 f_plt.pop().remove()
    #             while len(p_plt):
    #                 p_plt.pop().remove()
    #
    #             i += 1
