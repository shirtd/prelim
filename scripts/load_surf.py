import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from util.bottleneck import *
import numpy.linalg as la
import numpy as np
import os, sys

from util.plot import *
from util.data import *

VIEW = {'default' : 'side',
        'side' : {  'view' : (-11.800830502323867, 80.88795919149756, 9.035877007511106,
                        np.array([-1.00787402,  1.01587307,  0.6490444])),
                    'zoom' : 1.6, 'roll' : -89},
        'top' : {   'view' : (0.0, 0.0, 5.79555495773441, np.array([0. , 0. , 0.5])),
                    'zoom' : 1.6, 'roll' : 0}}#-80}}

def tup_d(s, t):
    return np.sqrt((s[0] - t[0])**2 + (s[1] - t[1]) ** 2)

def edges_to_path(E):
    E = [[tuple(p) for p in e] for e in E]
    adj = {}
    for p,q in E:
        if not p in adj:
            adj[p] = set()
        if not q in adj:
            adj[q] = set()
        adj[p].add(q)
        adj[q].add(p)
    V = {p for p in adj}
    paths = []
    while len(V):
        p = V.pop()
        path = [p]
        nbrs = adj[p].intersection(V)
        while len(nbrs):
            p = nbrs.pop()
            path.append(p)
            V.remove(p)
            nbrs = adj[p].intersection(V)
        paths.append(path)
    return [p + [p[0]] if tup_d(p[0],p[-1]) < 5 else p for p in paths]
    # _path = paths.pop()
    # while len(paths):
    #     i00 = min([(i, tup_d(p[0], _path[0])) for i,p in enumerate(paths)], key=lambda i_d: i_d[1])
    #     i01 = min([(i, tup_d(p[0], _path[-1])) for i,p in enumerate(paths)], key=lambda i_d: i_d[1])
    #     i10 = min([(i, tup_d(p[-1], _path[0])) for i,p in enumerate(paths)], key=lambda i_d: i_d[1])
    #     i11 = min([(i, tup_d(p[-1], _path[-1])) for i,p in enumerate(paths)], key=lambda i_d: i_d[1])
    #     imm = min([(0, i00), (1, i01), (2, i10), (3, i11)], key=lambda t_i_d: t_i_d[1][1])
    #     p = paths.pop(imm[1][0])
    #     if imm[0] == 0:
    #         _path = p[::-1] + _path
    #     elif imm[0] == 1:
    #         _path = _path + p
    #     elif imm[0] == 2:
    #         _path = p + _path
    #     elif imm[0] == 3:
    #         _path = _path + p[::-1]
    # return _path + [_path[0]]

class FeaturePlot:
    def __init__(self, q, barcode, plot_simplices=True, chain_color=COLOR['red'], rad=0.002):
        paths = edges_to_path(q['chain'][1])
        chains = [np.array([[Y[a,b], X[a,b], barcode.G[a,b]] for a,b in p]) for p in paths]
        birth = np.array([[Y[a,b], X[a,b], barcode.G[a,b]] for a,b in q['birth'][1]])
        death = np.array([[Y[a,b], X[a,b], barcode.G[a,b]] for a,b in q['death'][1] + [q['death'][1][0]]])
        self.pts = np.vstack([c[:-1] for c in chains])
        self.centroid = self.pts.sum(axis=0) / len(self.pts)
        self.radius = max(la.norm(self.centroid - p) for p in self.pts)
        self.elements = [self.plot(c, rad, chain_color) for c in chains]
        if plot_simplices:
            self.elements.append(self.plot(birth, 0.0025, (0,1,0)))
            self.elements.append(self.plot(death, 0.002, (0,0,1)))
    def plot(self, curve, radius, color):
        c = plot3d(curve[:,0], curve[:,1], curve[:,2])
        c.parent.parent.filter.radius = radius
        c.actor.property.lighting = False
        c.actor.property.color = color
        return c
    def remove(self):
        for e in self.elements:
            e.parent.parent.parent.parent.remove()

class Interact:
    def __init__(self, fig, ax, barcode, dat, surf):
        self.fig, self.ax = fig, ax
        self.barcode, self.dat = barcode, dat
        self.surf = surf
        self.interact = True
        self.jmap = ['res', 'cuts', 'rel']
        self.names = ['A', 'B', 'C', 'D', 'E']
        self.feature_plt = []
        self.pt_plt = []
        self.queries = []
        self.connect()
    def focus(self, elem):
        self.surf.cam.focal_point = elem.centroid
        self.surf.cam.zoom(1/(1.5*elem.radius))
    def connect(self):
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.cid = None
    def get_axis(self, event):
        for i in range(self.ax.shape[0]):
            for j in range(self.ax.shape[1]):
                if event.inaxes == self.ax[i,j]:
                    return i,j
        return None
    def clear(self):
        while len(self.feature_plt):
            self.feature_plt.pop().remove()
        while len(self.pt_plt):
            self.pt_plt.pop().remove()
    def plot_query(self, q, i, j, plot_pt=True, plot_simplices=True, chain_color=COLOR['red'], rad=0.002):
        for l, nm in enumerate(self.names):
            self.surf['cut'][nm]['visible'] = False if l <= i else True
        self.feature_plt.append(FeaturePlot(q, self.barcode, plot_simplices, chain_color,rad))
        if plot_pt:
            self.pt_plt.append(self.ax[i,j].scatter(q['pt'].birth, q['pt'].death,
                                s=15, color=COLOR['red'], marker='^', zorder=3, alpha=1))
            # for l, ax in enumerate(self.ax[i]):
            #     self.pt_plt.append(ax.scatter(q['pt'].birth, q['pt'].death, s=10, color=COLOR['red'],#'black',
            #                                 marker='^', zorder=3, alpha=0.3 if l!=j else 1.))
        plt.pause(0.1)
    def query(self, b, d, i, j, plot_pt=True, plot_simplices=True, chain_color=COLOR['red'], rad=0.002):
        q = self.barcode.query(self.dat[self.jmap[j]][i], b, d)
        self.queries.append(q)
        self.plot_query(q, i, j, plot_pt, plot_simplices, chain_color, rad)
    def onclick(self, event):
        ij = self.get_axis(event)
        if self.interact and ij is not None:
            self.clear()
            self.query(event.xdata, event.ydata, *ij)


if __name__ == "__main__":
    fname = sys.argv[1]
    self = load_barcodes(fname)
    G = self.field
    X_RNG = np.linspace(-1,1,self.N)
    Y_RNG = np.linspace(-1,1,self.N)
    X, Y = np.meshgrid(X_RNG, Y_RNG)

    SURF_ARGS = {   'A' : {'min' : 0, 'max' : self.cuts[0], 'color' : COLOR['green'],'opacity' : 0.5},
                    'B' : {'min' : self.cuts[0], 'max' : self.cuts[1], 'color' : COLOR['blue'], 'opacity' : 0.5},
                    'C' : {'min' : self.cuts[1], 'max' : self.cuts[2], 'color' : COLOR['purple'], 'opacity' : 0.5},
                    'D' : {'min' : self.cuts[2], 'max' : self.cuts[3], 'color' : COLOR['yellow'],  'opacity' : 0.5},
                    'E' : {'min' : self.cuts[3], 'max' : 1., 'color' : COLOR['gray'],  'opacity' : 0.5}}

    CONT_ARGS = {   'A_c' : {'scalar' : [self.cuts[0]], 'color' : COLOR['green']},
                    'B_c' : {'scalar' : [self.cuts[1]], 'color' : COLOR['blue']},
                    'C_c' : {'scalar' : [self.cuts[2]], 'color' : COLOR['purple']},
                    'D_c' : {'scalar' : [self.cuts[3]], 'color' : COLOR['yellow']}}

    DIM = 0
    SAVE = False

    surf = SurfacePlot(X, Y, G, SURF_ARGS, CONT_ARGS, VIEW)

    surf.reset_view('top')

    CMAP = [(v['color'], (v['min'], v['max'])) for v in SURF_ARGS.values()]

    plt.ion()
    fig, ax = init_dgms(len(self.cuts)-1)

    barcode = self.barcodes[-1]

    THRESH = 4 * np.sqrt(2 * (2 / self.N) ** 2)

    dat = barcode.get_dio_wrap(DIM, THRESH)

    # kwmap = {0 : {'alpha' : 0.1, 'size' : 2, 'zorder' : 1},
    #             1 : {'alpha' : 0.1, 'size' : 2, 'zorder' : 1},
    #             2 : {'alpha' : 0.5, 'size' : 2, 'zorder' : 2}}

    kwmap = {0 : {'alpha' : 0.5, 'size' : 4, 'zorder' : 1},
                1 : {'alpha' : 0.5, 'size' : 4, 'zorder' : 1},
                2 : {'alpha' : 0.5, 'size' : 4, 'zorder' : 2}}

    ax_rows = [[(ax[i,j], plot_diagram(ax[i,j], dat[k][i].dgm_np, CMAP, THRESH, **kwmap[j]))
                for j,k in enumerate(['res', 'cuts', 'rel'])] for i in range(len(self.cuts))]

    for i in range(len(self.cuts)):
        plot_diagram(ax[i,1], self.full_barcode[DIM], CMAP, THRESH, False, alpha=0.1, size=3, zorder=1)

    # matches = []
    # for i in range(len(self.cuts)):
    #     bad_match = dat['cuts'][i].match_death(dat['res'][i])
    #     no_match = dat['cuts'][i].unmatched_death(dat['res'][i])
    #     matches.append((bad_match, no_match))
    #     dat['cuts'][i].set_prio([p for (p,_,_),_ in bad_match] + [p for (p,_,_) in no_match])
    #     dat['res'][i].set_prio([p for _,(p,_,_) in bad_match])
    #
    #     dgm = [(p.birth, p.death) for (p,_,_),_ in bad_match]
    #     dgm_s = [(p.birth, p.death) for _,(p,_,_) in bad_match]
    #     dgm_n = [(p.birth, p.death) for (p,_,_) in no_match]
    #
    #     plot_diagram(ax[i,1], dgm, CMAP, plot_diag=False, alpha=1., size=5, zorder=2)
    #     plot_diagram(ax[i,0], dgm_s, CMAP, plot_diag=False, alpha=1., size=5, zorder=2)
    #     plot_diagram(ax[i,1], dgm_n, CMAP, plot_diag=False, alpha=1., size=5, zorder=2, marker='^')

    unmatches = []
    for i in range(len(self.cuts)):
        no_match = dat['res'][i].unmatched_death(dat['cuts'][i])
        unmatches.append(no_match)
        dat['res'][i].set_prio([p for (p,_,_) in no_match])
        dgm_n = [(p.birth, p.death) for (p,_,_) in no_match]
        plot_diagram(ax[i,0], dgm_n, CMAP, plot_diag=False, alpha=1., size=10, zorder=2, marker='^')

    inter = Interact(fig, ax, barcode, dat, surf)

    if SAVE:
        inter.interact = False

        di = 0
        name = os.path.splitext(os.path.basename(fname))[0]
        dir = os.path.join('figures', 'experiments', 'matching')#, name)
        while os.path.exists(os.path.join(dir, '%s-%d' % (name, di))):
            di += 1
        dir = os.path.join(dir, '%s-%d' % (name, di))
        print('creating directory %s' % dir)
        os.makedirs(dir)

        plt.savefig(os.path.join(dir, 'full-dgm.png'), dpi=300)
        surf.save(os.path.join(dir, 'full-surf_top.png'), (2000, 2000))

        vw = view()
        view(vw[0], 60, vw[2], vw[3])
        surf.save(os.path.join(dir, 'full-surf_side.png'), (2000, 2000))
        surf.reset_view('top')

        for i, unmatch in enumerate(unmatches):
            # l = self.cuts[i] - 0.03
            # ax[i,0].set_xlim(l, l + 0.15)
            # ax[i,0].set_ylim(l, l + 0.15)
            for j, (p,_,_) in enumerate(unmatch):
                inter.query(p.birth, p.death, i, 0, True, False, COLOR['red'], 0.003)
                plt.pause(0.1)
                plt.savefig(os.path.join(dir, 'unmatch-dgm-%d_%d.png' % (i, j)), dpi=300)
                surf.reset_view('top')
                surf.save(os.path.join(dir, 'unmatch-surf-%d_%d.png' % (i, j)), (4000, 4000))
                inter.focus(inter.feature_plt[0])
                surf.save(os.path.join(dir, 'unmatch-surf_zoom-%d_%d.png' % (i, j)), (1000, 1000))
                vw = view()
                view(vw[0], 60, vw[2], vw[3])
                surf.save(os.path.join(dir, 'umatch-surf_zoom_side-%d_%d.png' % (i, j)), (1000, 1000))
                # input('...')
                inter.clear()
                surf.reset_view('top')

        # for i, (match, unmatch) in enumerate(matches):
        #     l = self.cuts[i] - 0.03
        #     ax[i,0].set_xlim(l, l + 0.15)
        #     ax[i,0].set_ylim(l, l + 0.15)
        #     for j, ((p,_,_),(q,_,_)) in enumerate(match):
        #         inter.query(q.birth, q.death, i, 0, False, False, (1,0,0),rad=0.0015)
        #         inter.query(p.birth, p.death, i, 1, True, False)
        #         inter.pt_plt = inter.pt_plt + ax[i,0].plot([p.birth, q.birth],[p.death, q.death], c='black', alpha=0.25, zorder=2, linestyle='dotted')
        #         plt.pause(0.1)
        #         plt.savefig(os.path.join(dir, 'match-dgm-%d_%d.png' % (i, j)), dpi=300)
        #         surf.reset_view('top')
        #         surf.save(os.path.join(dir, 'match-surf-%d_%d.png' % (i, j)), (4000, 4000))
        #         inter.focus(inter.feature_plt[0])
        #         surf.save(os.path.join(dir, 'match-surf_zoom-%d_%d.png' % (i, j)), (1000, 1000))
        #         vw = view()
        #         view(vw[0], 60, vw[2], vw[3])
        #         surf.save(os.path.join(dir, 'match-surf_zoom_side-%d_%d.png' % (i, j)), (1000, 1000))
        #         # input('...')
        #         inter.clear()
        #     for j, (p,_,_) in enumerate(unmatch):
        #         inter.query(p.birth, p.death, i, 1)
        #         plt.pause(0.1)
        #         plt.savefig(os.path.join(dir, 'unmatch-dgm-%d_%d.png' % (i, j)), dpi=300)
        #         surf.reset_view('top')
        #         surf.save(os.path.join(dir, 'unmatch-surf-%d_%d.png' % (i, j)), (4000, 4000))
        #         inter.focus(inter.feature_plt[0])
        #         surf.save(os.path.join(dir, 'unmatch-surf_zoom-%d_%d.png' % (i, j)), (1000, 1000))
        #         vw = view()
        #         view(vw[0], 60, vw[2], vw[3])
        #         surf.save(os.path.join(dir, 'umatch-surf_zoom_side-%d_%d.png' % (i, j)), (1000, 1000))
        #         # input('...')
        #         inter.clear()
        #         surf.reset_view('top')
