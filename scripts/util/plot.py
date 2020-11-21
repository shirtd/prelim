import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from mayavi.mlab import *
from mayavi import mlab
from mayavi.modules.surface import Surface
from util.data import COLOR
import dionysus as dio

def init_bcode(ncuts):
    ht = ncuts+1
    fig, ax = plt.subplots(ht, 3, sharex=True, sharey=True, figsize=(11, ht*2))
    for a in ax:
        for aa in a:
            # aa.set_xlim(-0.1, 1.1)
            # aa.set_ylim(-0.2, 1.2)
            aa.get_yaxis().set_visible(False)
            aa.get_xaxis().set_visible(False)

    ax[0,0].set_title(r"$\mathrm{H}_0$")
    ax[0,1].set_title(r"$\mathrm{H}_1$")
    ax[0,2].set_title(r"$\mathrm{H}_2$")
    plt.tight_layout()
    return fig, ax

def plot_barcode(axis, dgm, cmap, lw=5, super=False):
    for i, (birth, death) in enumerate(dgm):
        i = 1 - i / len(dgm)
        # i = (birth + death) / 2
        infty = np.inf
        if super:
            infty *= -1
            death, birth = birth,death
        for color, (a,b) in cmap:#.items():
            if a < birth and death <= b:
                axis.plot([birth, death], [i, i], lw=lw, c=color)
            elif birth < a and death > a and death <= b:
                axis.plot([a, death], [i, i], lw=lw, c=color)
            elif birth > a and birth < b and death > b:
                axis.plot([birth, b], [i, i], lw=lw, c=color)
            elif birth <= a and b < death:
                axis.plot([b, a], [i, i], lw=lw, c=color)
        if death == np.inf:
            axis.plot([1., 1.1], [i, i], c='black', linestyle='dotted',  lw=lw/2)
        # ax.set_ylim(-1, 4)
        axis.get_yaxis().set_visible(False)

def plot_barcodes(axs, dgms, cmap, lw=5, super=False, clear=False):
    for i, ax in enumerate(axs):
        if clear:
            ax.cla()
            ax.set_xlim(-0.1, 1.1)
            # ax.set_ylim(-0.2, 1.2)
        plot_barcode(ax, dgms[i], cmap, lw, super)

def plot_multi_bc(axss, dgmss, cmap, lw=5, super=False, clear=False):
    for i,axs in enumerate(axss):
        plot_barcodes(axs, dgmss[i], cmap, lw, super, clear)

def init_dgms(ncuts):
    ht = ncuts+1
    # fig, ax = plt.subplots(ht, 3, sharex=True, sharey=True, figsize=(11, ht*3))
    fig, ax = plt.subplots(ht, 3, sharey=True, figsize=(11, ht*3))

    # ax[0,0].set_title(r"$\mathrm{H}_0$")
    # ax[0,1].set_title(r"$\mathrm{H}_1$")
    # ax[0,2].set_title(r"$\mathrm{H}_2$")

    # ax[0,0].set_title(r"$\mathrm{H}_0$")
    # ax[0,1].set_title(r"$\mathrm{H}_1$")
    # ax[0,2].set_title(r"$\mathrm{H}_2$")

    plt.tight_layout()
    return fig, ax

def get_cut_index(pt, cuts):
    if isinstance(pt, dio.DiagramPoint):
        pt = [pt.birth, pt.death]
    for i, (a,b) in enumerate(zip([0] + cuts[:-1], cuts)):
        if a <= pt[0] and pt[0] < b:
            return i
    return len(cuts)

def get_color(pt, cmap):
    if isinstance(pt, dio.DiagramPoint):
        pt = [pt.birth, pt.death]
    for color, (a,b) in cmap:
        if a <= pt[0] and pt[0] < b:
            return color
    return (0,0,0)

def plot_diagram(axis, dgm, cmap, thresh=-np.inf, plot_diag=True, alpha=0.5, size=5, zorder=1, **kw):
    # lim = max(max(p) for p in dgm) if lim is None else lim
    lim = max(max(c[1]) for c in cmap)
    dgm = np.array([[b, d if d < np.inf else lim*1.2]for b,d in dgm if d - b > thresh]) if len(dgm) else np.ndarray((0,2))
    if plot_diag:
        axis.plot([0,lim*1.1], [0,lim*1.1], c='black', alpha=0.5, zorder=0)
        axis.plot([0,lim*1.1], [lim*1.1,lim*1.1], c='black', alpha=0.5, zorder=0, linestyle='dotted')
        axis.plot([lim*1.1,lim*1.1], [lim*1.1,1.2*lim], c='black', alpha=0.5, zorder=0, linestyle='dotted')
    color = [get_color(pt, cmap) for pt in dgm]
    axis.scatter(dgm[:,0], dgm[:,1], color=color, s=size, alpha=alpha, zorder=zorder, **kw)
    axis.autoscale(False)
    # for i, (birth, death) in enumerate(dgm):
    #     for color, (a,b) in cmap:#.items():
    #         if a <= birth < b:
    #             axis.scatter(birth, death, s=10, alpha=0.5, zorder=2, color=color)
    #         if a <= death < b:
    #             axis.scatter(birth, death, s=10, marker='D', alpha=0.5, zorder=1, color=color)
    return dgm

def plot_diagrams(axs, dgms, cmap, thresh, clear=False):
    ax_cols = []
    for i, ax in enumerate(axs):
        if clear:
            ax.cla()
            # ax.set_ylim(-0.2, 1.2)
        ax_cols.append((ax, plot_diagram(ax, dgms[i], cmap, thresh)))
    return ax_cols

def plot_multi_dgm(axss, dgmss, cmap, thresh, clear=False):
    ax_rows = []
    for i,axs in enumerate(axss):
        ax_rows.append(plot_diagrams(axs, dgmss[i], cmap, thresh, clear))
    return ax_rows

class SurfaceElement:
    def __init__(self, ctl, name):
        self._o = Surface()
        self._o.name = name
        self._o.enable_contours = True
        self._o.actor.property.lighting = False
        self._o.actor.mapper.scalar_visibility = False
        ctl.add_child(self._o)
        self._props = {'visible' : ['visible'],
                        'color' : ['actor', 'property', 'color'],
                        'opacity' : ['actor', 'property', 'opacity'],
                        'backface_culling' : ['actor', 'property', 'backface_culling']}
    def _init_props(self, **kwargs):
        for k,v in kwargs.items():
            self[k] = v
    def _trait_search(self, l, set=None, p=None):
        if len(l) > 1:
            p = (self._o if p is None else p).trait_get(l[0])[l[0]]
            return self._trait_search(l[1:], set, p)
        elif set is not None:
            (self._o if p is None else p).trait_set(**{l[0] : set})
            return set
        else:
            return p.trait_get(l[0])[l[0]]
    def __getitem__(self, key):
        return self._trait_search(self._props[key])
    def __setitem__(self, key, val):
        return self._trait_search(self._props[key], val)

class SurfaceCut(SurfaceElement):
    def __init__(self, ctl, name, **kwargs):
        SurfaceElement.__init__(self, ctl, name)
        self._o.contour.filled_contours = True
        self._o.actor.property.opacity = 0.5
        self._props = {'min' : ['contour', 'minimum_contour'],
                        'max' : ['contour', 'maximum_contour'],
                        **self._props}
        self._init_props(**kwargs)

class SurfaceContour(SurfaceElement):
    def __init__(self, ctl, name, **kwargs):
        SurfaceElement.__init__(self, ctl, name)
        self._o.contour.filled_contours = False
        self._o.contour.auto_contours = False
        self._o.actor.property.line_width = 4
        self._o.visible = False
        self._props = {'scalar' : ['contour', 'contours'], **self._props}
        self._init_props(**kwargs)

class SurfacePlot:
    def __init__(self, X, Y, G, cuts, contours, view):
        self.s0 = surf(X.T, Y.T, G)
        self.s0.visible = False
        self.ctl = self.s0.parent
        self.gcf = gcf()
        self.scene = self.gcf.scene
        self.cam = self.scene.camera
        self.scene.parallel_projection = True
        self.scene.background = (1,1,1)
        self._elem = {'cut' : {}, 'cont' : {}}
        for k, v in cuts.items():
            self['cut'][k] = SurfaceCut(self.ctl, k, **v)
        # self._cont_elem = {}
        for k, v in contours.items():
            self['cont'][k] = SurfaceContour(self.ctl, k, **v)
        self._view = view
        self.reset_view(self._view['default'])
    def __getitem__(self, key):
        return self._elem[key]
    def __setitem__(self, key, val):
        self._elem[key] = val
    def reset_view(self, key):
        self.set_view(**self._view[key])
    def set_view(self, view=None, zoom=None, roll=None):
        if view is not None:
            mlab.view(*view)
        if zoom is not None:
            self.cam.parallel_scale = zoom
        if roll is not None:
            mlab.roll(roll)
    def focus_low(self, name):
        self.focus_scalar(name)
        for k, v in self._elem['cut'].items():
            c = '%s_c' % k
            if v['max'] == self['cut'][name]['min']:
                v['opacity'] = 0.1
                v['visible'] = True
                if c in self['cont']:
                    self['cont'][c]['visible'] = True
            else:
                v['opacity'] = 0.5
                if c in self['cont']:
                    self['cont'][c]['visible'] = False
    def focus_high(self, name):
        self.focus_scalar(name)
        for k, v in self._elem['cont'].items():
            if v['scalar'][0] == self['cut'][name]['min']:
                v['visible'] = True
            else:
                v['visible'] = False
    def focus_scalar(self, name):
        for k, v in self._elem['cut'].items():
            v['opacity'] = 0.5
            if v['max'] <= self['cut'][name]['min']:
                v['visible'] = False
            else:
                v['visible'] = True
    def save(self, name, size=(1500, 868)):
        # print('saving %s' % name)
        mlab.savefig(name, size=size)
