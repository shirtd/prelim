import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from util.bottleneck import *
import numpy.linalg as la
import numpy as np
import os, sys

from util.plot import *
from util.data import *

from load_surf import VIEW, tup_d, edges_to_path

class FeaturePlot:
    def __init__(self, pt, dat, X, Y, G, chain_color=COLOR['red'], rad=0.007):
        c_idx = [c.index for  c in dat.hom[dat.pair(pt)]]
        C = [[grid_coord(v, len(X)) for v in dat.filt[c]] for c in c_idx]
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

    PLOT_SURF = False


    # CMAP = [(COLOR['orange'], (0, self.cuts[0])),
    #         (COLOR['green'], (self.cuts[0], self.cuts[1]))
    CMAP = [(COLOR['green'],    (0, self.cuts[0])),
            (COLOR['blue'],     (self.cuts[0], self.cuts[1])),
            (COLOR['purple'],   (self.cuts[1], self.cuts[2])),
            (COLOR['yellow'],   (self.cuts[2], 1))]
            # (COLOR['yellow'],   (self.cuts[2], self.cuts[3])),
            # (COLOR['gray'],     (self.cuts[3], 1.))]

    LABELS = ['A', 'B', 'C', 'D']#, 'E']

    SURF_ARGS = {l : {'min' : a, 'max' : b, 'color' : c, 'opacity' : 0.5} for l,(c,(a,b)) in zip(LABELS, CMAP)}
    CONT_ARGS = {'_'.join((l,'c')) : {'scalar' : [b], 'color' : c} for l,(c,(a,b)) in zip(LABELS, CMAP)}

    # SURF_ARGS = {   'A' : {'min' : 0, 'max' : self.cuts[0], 'color' : COLOR['green'],'opacity' : 0.5},
    #                 'B' : {'min' : self.cuts[0], 'max' : self.cuts[1], 'color' : COLOR['blue'], 'opacity' : 0.5},
    #                 'C' : {'min' : self.cuts[1], 'max' : self.cuts[2], 'color' : COLOR['purple'], 'opacity' : 0.5},
    #                 'D' : {'min' : self.cuts[2], 'max' : self.cuts[3], 'color' : COLOR['orange'],  'opacity' : 0.5},
    #                 'E' : {'min' : self.cuts[3], 'max' : 1., 'color' : COLOR['gray'],  'opacity' : 0.5}}
    #
    # CONT_ARGS = {   'A_c' : {'scalar' : [self.cuts[0]], 'color' : COLOR['green']},
    #                 'B_c' : {'scalar' : [self.cuts[1]], 'color' : COLOR['blue']},
    #                 'C_c' : {'scalar' : [self.cuts[2]], 'color' : COLOR['purple']},
    #                 'D_c' : {'scalar' : [self.cuts[3]], 'color' : COLOR['orange']}}

    if PLOT_SURF:
        surf = SurfacePlot(X, Y, G, SURF_ARGS, CONT_ARGS, VIEW)
        surf.reset_view('top')

    THRESH = 4 * np.sqrt(2 * (2 / self.N) ** 2)

    barcode = self.barcodes[-1]

    plt.ion()

    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(15, 4))

    # ax[0,0].set_title(r"$\mathrm{H}_0$")
    ax[0].set_title(r"$\mathrm{H}_1$ full")
    ax[1].set_title(r"$\mathrm{H}_1$ restricted ($\omega = %0.1f$)" % self.cuts[0])
    ax[2].set_title(r"$\mathrm{H}_1$ restricted ($\omega = %0.1f$)" % self.cuts[1])
    ax[3].set_title(r"$\mathrm{H}_1$ restricted ($\omega = %0.1f$)" % self.cuts[2])

    plt.tight_layout()
    plot_diagram(ax[0], self.full_barcode[1], CMAP, THRESH, alpha=0.75, size=10, zorder=1)
    plot_diagram(ax[1], barcode.dgms_res[0][1], CMAP, THRESH, alpha=0.75, size=10, zorder=1)
    plot_diagram(ax[2], barcode.dgms_res[1][1], CMAP, THRESH, alpha=0.75, size=10, zorder=1)
    plot_diagram(ax[3], barcode.dgms_res[2][1], CMAP, THRESH, alpha=0.75, size=10, zorder=1)

    dio_dat = barcode.get_dio_dat()
    dat = {'full' : [DioWrap(dio_dat['full'], dim, THRESH) for dim in range(3)],
                'cuts' : [[DioWrap(dio_dat['full'], dim, THRESH, cut) for dim in range(3)] for cut in self.cuts],
                'res' : [[DioWrap(d, dim, THRESH) for dim in range(3)] for d in dio_dat['res']],
                'rel' : [[DioWrap(d, dim, THRESH) for dim in range(3)] for d in dio_dat['rel']]}

    DIM = 1
    matchings = [dat['full'][DIM].match_death(r[DIM]) for r in dat['res']]

    di = 0
    name = os.path.splitext(os.path.basename(fname))[0]
    dir = os.path.join('figures', 'experiments', 'matching')#, name)
    while os.path.exists(os.path.join(dir, '%s-%d' % (name, di))):
        di += 1
    dir = os.path.join(dir, '%s-%d' % (name, di))
    print('creating directory %s' % dir)
    os.makedirs(dir)
    
    plt.savefig(os.path.join(dir, 'full-dgm.pdf'), dpi=300)

    if PLOT_SURF:
        surf.save(os.path.join(dir, 'full-surf_top.png'), (2000, 2000))

        vw = view()
        view(vw[0], 80, vw[2], vw[3])
        surf.save(os.path.join(dir, 'full-surf_side.png'), (2000, 2000))
        surf.reset_view('top')

    ps = {p for matches in matchings for (p,_,_), _ in matches}
    e_plt, f_plt = [], []

    for j, _p in enumerate(ps):
        _ft = FeaturePlot(_p, dat['full'][DIM], X, Y, self.field, (0,0,0))
        # e_plt.append(ax[0].scatter(_p.birth, _p.death, s=10, color=(0,0,0), marker='D', zorder=3))
        e_plt.append(ax[0].scatter(_p.birth, _p.death, s=10, color=get_color(_p, CMAP), marker='D', zorder=3))
        f_plt.append(_ft)

        if PLOT_SURF:
            surf.reset_view('top')
            surf.save(os.path.join(dir, 'surf_top-%d.png' % j), (3000, 3000))
            vw = view()
            view(vw[0], 80, vw[2], vw[3])
            surf.save(os.path.join(dir, 'surf_side-%d.png' % j), (3000, 3000))
            surf.reset_view('top')

        for i, (axis, matches) in enumerate(zip(ax[1:], matchings)):
            if PLOT_SURF:
                surf['cut'][LABELS[i]]['opacity'] = 0.1
            for (p,_,_), (q,_,_) in matches:
                if p == _p:
                    # surf['cont']['%s_c' % LABELS[i]]['visible'] = True
                    e_plt.append(axis.scatter(p.birth, p.death, s=15, color=get_color(p, CMAP), alpha=0.25, marker='D', zorder=3))
                    e_plt.append(axis.scatter(q.birth, q.death, s=15, color=get_color(q, CMAP), zorder=3))
                    # e_plt.append(axis.scatter(p.birth, p.death, s=10, color=(0,0,0), alpha=0.5, marker='D', zorder=3))
                    # e_plt.append(axis.scatter(q.birth, q.death, s=10, color=COLOR['red'], zorder=3))
                    prev, cut = p.birth+0.025, get_cut_index(p, self.cuts)
                    while cut < len(self.cuts) and q.birth > self.cuts[cut]:
                        e_plt = e_plt + axis.plot([prev, self.cuts[cut]],[p.death, q.death],
                                        color=CMAP[cut][0], alpha=0.25, zorder=2, linestyle='dotted')
                        cut += 1
                    e_plt = e_plt + axis.plot([self.cuts[cut-1], q.birth],[p.death, q.death],
                                        color=CMAP[cut][0], alpha=0.25, zorder=2, linestyle='dotted')

                    f_plt.append(FeaturePlot(q, dat['res'][i][DIM], X, Y, barcode.G, COLOR['red']))

                    if PLOT_SURF:
                        surf.reset_view('top')
                        surf.save(os.path.join(dir, 'surf_top-%d_%d.png' % (j, i)), (3000, 3000))
                        vw = view()
                        view(vw[0], 80, vw[2], vw[3])
                        surf.save(os.path.join(dir, 'surf_side-%d_%d.png' % (j, i)), (3000, 3000))
                        surf.reset_view('top')
                        f_plt.pop().remove()

                    # surf['cont']['%s_c' % LABELS[i]]['visible'] = False


        plt.pause(0.1)
        plt.savefig(os.path.join(dir, 'dgm-%d.pdf' % j), dpi=300)

        while len(f_plt):
            f_plt.pop().remove()

        while len(e_plt):
            e_plt.pop().remove()

        if PLOT_SURF:
            for l in LABELS:
                surf['cut'][l]['opacity'] = 0.5
