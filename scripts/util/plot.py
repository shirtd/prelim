import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mayavi.mlab import *
from mayavi import mlab
from mayavi.modules.surface import Surface

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
    def save(self, name):
        print('saving %s' % name)
        mlab.savefig(name, size=(1500, 868))
