from sim.util import stuple, pmap
from scipy.spatial import distance_matrix
from sklearn.neighbors import KernelDensity
from itertools import combinations
from functools import reduce
from heapq import merge
from tqdm import tqdm
import numpy as np
import bisect

class Simplex(tuple):
    def __new__(cls, vertices, data=None, relative=False, values={}):
        return tuple.__new__(cls, stuple(vertices))
    def __init__(self, vertices, data=None, relative=False, values={}):
        tuple.__init__(self)
        self.dim = len(self) - 1
        self.data = data
        self.values = values
        self.identify = None
        self.offset = None
        self.identified = set()
        # self.cofaces = {}
        self.faces = []
        self.relative = relative
    def set_faces(self, faces):
        self.faces = faces
        # for f in faces:
        #     f.cofaces[self] = self
    def star(self, i):
        return [(i,) + f for f in self.faces] if self.dim > 0 else [(i,)]
    def face_it(self):
        for i in range(self.dim+1):
            yield stuple(self[:i]+self[i+1:]) # Simplex(self[:i]+self[i+1:])
    def set_index(self, i):
        self.index = i
        return self
    def __contains__(self, s) -> bool:
        if isinstance(s, Simplex):
            return all(v in self for v in s)
        elif isinstance(s, int):
            return tuple.__contains__(self, s)
        return False
    def __lt__(self, other):
        if (self.data is None or other.data is None
            or self.data == other.data):
            return tuple.__lt__(self, other)
        return self.data < other.data
    def __repr__(self):
        if self.data is None:
            return tuple.__repr__(self)
        if not len(self.values):
            return '%s:%0.4f' % (tuple.__repr__(self), self.data)
        return '%s:%0.4f:%s' % (tuple.__repr__(self), self.data, str(self.values))

# class BiSimplex(Simplex):
#     def __new__(cls, vertices, data, relative=False, values={}):
#         return Simplex.__new__(cls, stuple(vertices), data)
#     def __init__(self, vertices, data, relative=False, values={}):
#         Simplex.__init__(self, vertices, data, relative)
#         self.values = values
#     # def swap_value(self):
#     #     return BiSimplex([v for v in self], self.value, self.data)
#     def __repr__(self):
#         if self.data is None:
#             return tuple.__repr__(self)
#         return '%s:%0.4f:%s' % (tuple.__repr__(self), self.data, str(self.values))


class SimplicialComplex:
    def __init__(self, dim):
        self.dim = dim
        # self.simplices = {d : set() for d in range(self.dim+1)}
    # def add(self, s):
    #     self.simplices[s.dim].add(s)
    # def remove(self, s):
    #     if s in self:
    #         self.simplices[s.dim].remove(s)
    # def __contains__(self, s):
    #     return s in self.simplices[s.dim]
    def chain_boundary(self, chain):
        bdy = set()
        for s in chain:
            for t in s.faces:
                if t in bdy:
                    bdy.remove(t)
                else:
                    bdy.add(t)
        return bdy

class FilteredSimplicialComplex(SimplicialComplex):
    def __init__(self, dim):
        SimplicialComplex.__init__(self, dim)
        self.filtration = []
        # self.boundary = []
        # self.imap = {}
        self.smap = {}
        # self.smap = {}
    def get_boundary(self, key=None, relative=False, sub=False):
        return self.boundary
    def get_value(self, i, key=None, sub=False):
        return self.get_simplex(i, key, sub).data
    def get_simplex(self, i, key=None, sub=False):
        return self.filtration[i]
    def __iter__(self):
        for s in self.filtration:
            yield s
    # def get_faces(self, s):
    #     return {self.imap[t] for t in s.faces} if s.dim > 0 else set()
    def add(self, s):
        # self.imap[s] = len(self.filtration)
        # self.boundary.append(self.get_faces(s))
        s.index = len(self.filtration)
        self.filtration.append(s)
        return s
        # SimplicialComplex.add(self, s)
    # def make_boundary(self):
    #     self.imap = {s : i for i,s in enumerate(self.filtration)}
    #     self.boundary = [{self.imap[t] for t in s.faces} if s.dim > 0 else set() for s in self.filtration]
    def sort_perm(self, key):
        idx = sorted(range(len(self)), key=lambda i: self[i].data)
        perm = {j : i for i,j in enumerate(idx)}
        return idx, perm
    # def sort(self, fsort=None, *args, **kw):
    def sort(self, key=None):
        # if fsort is None:
        #     fsort = lambda i: self[i].data
        # idx = sorted(range(len(self)), key=fsort)
        # perm = {j : i for i,j in enumerate(idx)}
        idx, perm = self.sort_perm(key)
        self.filtration = [self[i].set_index(j) for j,i in enumerate(idx)]
        # self.boundary = [{t.index for t in s.faces} for s in self]
        # filtration = []
        # for i,j in enumerate(idx):
        # self.filtration.sort()
        # self.imap = {s : perm[i] for s,i in self.imap.items()}
        # self.boundary = [{perm[j] for j in self.boundary[i]} for i in idx]
        # self.filtration.sort()
        # self.make_boundary()
    def get_boundary(self, *args, **kw):
        return [{t.index for t in s.faces} for s in self]
    def __len__(self):
        return len(self.filtration)
    def __getitem__(self, i):
        return self.filtration[i]


class RipsComplex(FilteredSimplicialComplex):
    def __init__(self, points, eps, dim, sort=True):
        FilteredSimplicialComplex.__init__(self, dim)
        self.points = points
        self.eps = eps
        self.n_pts = len(self.points)
        self.dmat = distance_matrix(self.points, self.points)
        self.lower_nbrs = {}
        self.smap = {}
        for i in range(self.n_pts):
            self.lower_nbrs[i] = self.get_lower_nbrs(i)
            self.add_cofaces(self.make_vertex(i), self.lower_nbrs[i])
        if sort:
            self.sort()
    def get_points(self, chain, key=None):
        return np.vstack([self.points[s] for s in chain])
    def make_vertex(self, i, relative=False, values={}):
        values['rips'] = 0
        s = Simplex((i,), 0, relative, values)
        self.smap[s] = s
        return s
    def make_simplex(self, i, t, relative=False, values={}):
        try:
            faces = [self.smap[f] for f in t.star(i)] + [t]
        except KeyError:
            faces = [self.smap[stuple(f)] for f in t.star(i)] + [t]
        d = max(f.data for f in faces) if t.dim > 0 else self.dmat[i,t[0]]
        values['rips'] = d
        s = Simplex(t + (i,), max(t.data, d), relative, values)#d)
        # s.faces = faces
        s.set_faces(faces)
        self.smap[s] = s
        return s
    def get_lower_nbrs(self, i):
        return [j for j in range(i) if self.dmat[i,j] <= self.eps]
    def add_cofaces(self, t, N):#, F):
        self.add(t)
        if t.dim < self.dim:
            for i in N:
                inbr = self.lower_nbrs[i]
                s = self.make_simplex(i, t)
                M = [j for j in inbr if j in N]
                self.add_cofaces(s, M)
    # def update(self, points, sort=True):
    #     dmat_side = distance_matrix(self.points, points)
    #     dmat_corner = distance_matrix(points, points)
    #     self.points = np.vstack([self.points, points])
    #     self.dmat = np.vstack([np.hstack([self.dmat, dmat_side]),
    #                         np.hstack([dmat_side.T, dmat_corner])])
    #     n_pts_old, self.n_pts = self.n_pts, len(self.points)
    #     for i in range(n_pts_old, self.n_pts):
    #         self.lower_nbrs[i] = self.get_lower_nbrs(i)
    #         self.add_cofaces(self.make_vertex(i), self.lower_nbrs[i])#, True)
    #     if sort:
    #         self.sort()

class TorusRipsComplex(RipsComplex):
    def __init__(self, points, eps, dim, sort=True):
        init_points = points
        self.torus_points, self.torus_vmap, self.torus_idxs, self.torus_offs = self.make_torus(points, eps)
        RipsComplex.__init__(self, self.torus_points, eps, dim, sort)
        self.build_faces()
    def build_faces(self):
        for s in self.filtration:
            if s.dim > 0:
                faces = [self.smap[f] for f in s.face_it()]
                s.set_faces(faces)
    def get_points(self, chain, key=None):
        return np.vstack([self.points[s] + s.offset for s in chain])
    def get_lower_nbrs(self, i):
        return {j : self.dmat[i,j] for j in range(i) if self.dmat[i,j] <= self.eps}
    def add_cofaces(self, t, N):
        self.add(t)
        if t.dim < self.dim:
            for i, d in N.items():
                inbr = self.lower_nbrs[i]
                s = self.make_simplex(i, t, d)
                M = {j : max(inbr[j], jd) for j, jd in N.items() if j in inbr}
                self.add_cofaces(s, M)
    def add(self, s):
        if not s in self.smap:
            s.index = len(self.filtration)
            self.filtration.append(s)
            self.smap[s] = s
            return s
        elif s.data < self.smap[s].data:
            s.index = self.smap[s].index
            self.smap[s] = s
            self.filtration[s.index] = s
            return s
        return None
    def make_vertex(self, i, relative=False, values={}):
        j = self.torus_vmap[i]
        values['rips'] = 0
        s = Simplex((j,), 0, relative, values)
        s.offset = self.torus_offs[i]
        return s
    def make_simplex(self, i, t, d, relative=False, values={}):
        j = self.torus_vmap[i]
        d = max(t.data, d)
        values['rips'] = d
        s = Simplex(t + (j,), d, relative, values)
        l = s.index(j)
        if s.offset is not None:
            print(s)
        if l == 0:
            s.offset = np.vstack([self.torus_offs[i], t.offset])
        elif l == len(t):
            s.offset = np.vstack([t.offset, self.torus_offs[i]])
        else:
            s.offset = np.vstack([t.offset[:l], self.torus_offs[i], t.offset[l:]])
        # s.offset = np.vstack([t.offset, self.torus_offs[i]])
        return s
    def make_torus(self, points, eps):
        l = 2*np.pi
        n = len(points)
        tiles = np.array([[0,0],[l,0],[l,l],[0,l]])
        J = [i for i,p in enumerate(points) if p[0] <= eps and p[1] <= eps]
        I = [i for i,p in enumerate(points) if p[0] <= eps]
        K = [i for i,p in enumerate(points) if p[1] <= eps]
        rng = list(range(n))
        idxs = rng + I + J + K
        vmap = {j : i for j,i in enumerate(idxs)}
        offs = np.vstack([t for t,l in zip(tiles, [rng,I,J,K]) for _ in l])
        torus_points = np.vstack([points, points[I] + tiles[1],
                                            points[J] + tiles[2],
                                            points[K] + tiles[3]])
        return torus_points, vmap, idxs, offs

class ScalarRipsComplex(RipsComplex):
    def __init__(self, points, fun, eps, dim):
        perm = sorted(range(len(fun)),key=lambda i: fun[i])
        imap = {j : i for i,j in enumerate(perm)}
        # relative_idx = {imap[i] for i in relative_idx}
        points, self.fun = points[perm], [fun[i] for i in perm]
        # self.fun = fun
        RipsComplex.__init__(self, points, eps, dim, False)
        self.rips_idx, self.rips_perm = self.sort_perm('rips')
        # self.rips_idx = sorted(range(len(self)), key=lambda i: self[i].data)
        # self.rips_perm = {j : i for i,j in enumerate(self.rips_idx)}
        # # self.rips_boundary = None
    def get_boundary(self, key, relative=False, sub=False):
        if key == 'rips':
            return [{self.rips_perm[t.index] for t in self[i].faces} for i in self.rips_idx]
        elif key == 'scalar':
            return RipsComplex.get_boundary(self)
        raise Exception('unknown value key %s' % key)
    def get_value(self, i, key, sub=False):
        if key == 'rips':
            return self.get_simplex(i, key, sub).values['rips']
        elif key == 'scalar':
            return self.get_simplex(i, key, sub).values['scalar']
        raise Exception('unknown value key %s' % key)
    def get_simplex(self, i, key, sub=False):
        if key == 'rips':
            return self[self.rips_idx[i]]
        elif key == 'scalar':
            return self[i]
        raise Exception('unknown value key %s' % key)
    def make_vertex(self, i, relative=False):
        values = {'scalar' : self.fun[i]}
        return RipsComplex.make_vertex(self, i, relative, values)
    def make_simplex(self, i, t, relative=False):
        values = {'scalar' : max(t.values['scalar'], self.fun[i])}
        return RipsComplex.make_simplex(self, i, t, relative, values)
    def sort_perm(self, key):
        idx = sorted(range(len(self)), key=lambda i: self[i].values[key])
        perm = {j : i for i,j in enumerate(idx)}
        return idx, perm
    def sort(self, key):
        idx, perm = self.sort_perm(key)
        self.filtration = [self[i].set_index(j) for j,i in enumerate(idx)]
    # def update(self, points, fun, relative_idx):
    #     self.fun += fun
    #     RelativeRipsComplex.update(self, points, relative_idx, False)
    #     self.sort('scalar')
    #     self.rips_idx, self.rips_perm = self.sort_perm('rips')
    def chain_boundary(self, chain):
        bdy = set()
        for s in chain:
            if not any(v in self.relative_vertex_idx for v in s):
            # if not any(any(v in bidx for v in s) for bidx in self.border_idx):
                for t in s.faces:
                    if t in bdy:
                        bdy.remove(t)
                    else:
                        bdy.add(t)
        return bdy

    # def reduce_eps(self, eps):
    #     self.eps = eps
    #     self.lower_nbrs = {i : self.get_lower_nbrs(i) for i,_ in enumerate(self.data)}
    #     mini = np.inf
    #     for (i,j),S in self.edge_simplices.copy().items():
    #         if self.dmat[i,j] > self.eps:
    #             for s in S:
    #                 self.remove(s)
    #             del self.edge_simplices[(i,j)]
    #
    # def update_nbrs(self, N, i, j):
    #     inbr = self.lower_nbrs[i]
    #     return
        # if len(t) >= 2:
        #     for e in combinations(t, 2):
        #         e = stuple(e)
        #         if not e in self.edge_simplices:
        #             self.edge_simplices[e] = set()
        #         self.edge_simplices[e].add(t)


class TorusScalarRips(TorusRipsComplex, ScalarRipsComplex):
    def __init__(self, points, fun, eps, dim):
        # init_points = points
        # self.torus_points, self.torus_vmap, self.torus_idxs = self.make_torus(points, eps)
        # self.fun = [fun[i] for i in self.torus_idxs]
        # RipsComplex.__init__(self, self.torus_points, eps, dim, False)
        # self.build_faces()
        # # self.sort('rips')
        # self.rips_idx, self.rips_perm = self.sort_perm('rips')
        #
        init_perm = sorted(range(len(fun)),key=lambda i: fun[i])
        init_points, init_fun = points[init_perm], [fun[i] for i in init_perm]
        # init_points, init_fun = points, fun
        self.torus_points, self.torus_vmap, self.torus_idxs, self.torus_offs = self.make_torus(init_points, eps)
        self.fun = [init_fun[i] for i in self.torus_idxs]
        # torus_points, torus_vmap, torus_idxs = self.make_torus(init_points, eps)
        # torus_fun = [init_fun[i] for i in torus_idxs]
        # perm = [i for i in range(len(torus_fun))]
        # imap = {j : i for i, j in enumerate(perm)}
        # self.torus_points, self.fun = torus_points[perm], [torus_fun[i] for i in perm]
        # self.torus_vmap = {imap[j] : imap[i] for j,i in torus_vmap.items()}
        # self.torus_idxs = [imap[i] for i in torus_idxs]
        RipsComplex.__init__(self, self.torus_points, eps, dim, False)
        self.sort('scalar')
        # self.torus_points, self.points = self.points, init_points[init_perm]
        # self.torus_fun, self.fun = self.fun, [init_fun[i] for i in init_perm]
        self.n_pts = len(self.points)
        self.build_faces()
        self.rips_idx, self.rips_perm = self.sort_perm('rips')
    def make_vertex(self, i, relative=False):
        values = {'scalar' : self.fun[i]}
        return TorusRipsComplex.make_vertex(self, i, relative, values)
    def make_simplex(self, i, t, d, relative=False):
        values = {'scalar' : max(t.values['scalar'], self.fun[i])}
        return TorusRipsComplex.make_simplex(self, i, t, d, relative, values)
    def get_lower_nbrs(self, i):
        return TorusRipsComplex.get_lower_nbrs(self,i)
    def add_cofaces(self, t, N):
        return TorusRipsComplex.add_cofaces(self, t, N)
    def add(self, s):
        return TorusRipsComplex.add(self, s)
    def tcc(self, dgm):
        # return dgm.betti(2) == 1
        if dgm.betti(2) == 0:
            return False
        chains = [dgm.get_chain(pt) for pt in dgm.get_inf(2)]
        return len([c for c in chains if self.is_relative_chain(c)]) == 1

class RelativeRipsComplex(RipsComplex):
    def __init__(self, points, relative_idx, eps, dim, sort=True):
        self.relative_vertex_idx = set(relative_idx)
        self.sub_idx, self.sub_imap = [], {}
        RipsComplex.__init__(self, points, eps, dim, sort)
    def get_simplex(self, i, key=None, sub=False):
        i = self.sub_idx[i] if sub else i
        return RipsComplex.get_simplex(self, i, key)
    def is_relative_chain(self, chain):
        chain_boundary_pts = {i for e in self.chain_boundary(chain) for i in e}
        return len(chain_boundary_pts) != 0
        # return (len(chain_boundary_pts) != 0 and
        #     all(i in self.relative_vertex_idx for i in chain_boundary_pts))
    def is_relative(self, i, key=None):
        return self.get_simplex(i, key).relative
    def make_vertex(self, i):
        relative = i in self.relative_vertex_idx
        return RipsComplex.make_vertex(self, i, relative)
    def make_simplex(self, i, t):
        relative = t.relative and self.smap[(i,)].relative
        return RipsComplex.make_simplex(self, i, t, relative)
    def add(self, s):
        s = RipsComplex.add(self, s)
        if s.relative:
            self.sub_imap[s.index] = len(self.sub_idx)
            self.sub_idx.append(s.index)
        return s
    def sort(self, key=None):
        idx, perm = self.sort_perm(key)
        self.sub_idx = [perm[i] for i in sorted(self.sub_idx, key=lambda i: perm[i])]
        self.sub_imap = {j : i for i,j in enumerate(self.sub_idx)}
        self.filtration = [self[i].set_index(j) for j,i in enumerate(idx)]
    def get_boundary(self, key=None, relative=True, sub=False):
        if relative:
            return [{t.index for t in s.faces if not t.relative} for s in self]
        elif sub:
            return [{self.sub_imap[t.index] for t in self[i].faces} for i in self.sub_idx]
        return RipsComplex.get_boundary(self)
    def tcc(self, dgm, connected_components=1):
        if len(self.relative_vertex_idx) > 0:
            if dgm.betti(2) == 0:
                return False
            chains = [dgm.get_chain(pt) for pt in dgm.get_inf(2)]
            return len([c for c in chains if self.is_relative_chain(c)]) == connected_components
        else:
            return dgm.betti(2) == connected_components

class RelativeTorusRips(RelativeRipsComplex, TorusRipsComplex):
    def __init__(self, points, relative_idx, eps, dim, sort=True):
        init_points = points
        self.torus_points, self.torus_vmap, self.torus_idxs, self.torus_offs = self.make_torus(points, eps)
        RelativeRipsComplex.__init__(self, self.torus_points, relative_idx, eps, dim, sort)
        # RipsComplex.__init__(self, self.torus_points, eps, dim, sort)
        self.build_faces()
    def make_vertex(self, i):
        j = self.torus_vmap[i]
        relative = j in self.relative_vertex_idx
        return TorusRipsComplex.make_vertex(self, i, relative)
    def make_simplex(self, i, t, d):
        j = self.torus_vmap[i]
        relative = t.relative and self.smap[(j,)].relative
        return TorusRipsComplex.make_simplex(self, i, t, d, relative)
    def get_lower_nbrs(self, i):
        return TorusRipsComplex.get_lower_nbrs(self, i)
    def add_cofaces(self, t, N):
        return TorusRipsComplex.add_cofaces(self, t, N)
    def add(self, s):
        if not s in self.smap:
            s.index = len(self.filtration)
            if s.relative:
                self.sub_imap[s.index] = len(self.sub_idx)
                self.sub_idx.append(s.index)
            self.filtration.append(s)
            self.smap[s] = s
            return s
        elif s.data < self.smap[s].data:
            s.index = self.smap[s].index
            self.smap[s] = s
            self.filtration[s.index] = s
            return s
        return None
    # def add(self, s):
    #     s = TorusRipsComplex.add(self, s)
    #     if s is not None and s.relative:
    #         self.sub_imap[s.index] = len(self.sub_idx)
    #         self.sub_idx.append(s.index)
    #     return s
    def tcc(self, dgm, connected_components=1):
        return RelativeRipsComplex.tcc(self, dgm, connected_components)
    # def make_simplex(self, i, t):
    #     faces = [self.smap[f] for f in t.star(i)] + [t]
    #     d = max(f.data for f in faces) if t.dim > 0 else self.dmat[i,t[0]]
    #     relative = t.relative and self.smap[(i,)].relative
    #     s = Simplex(t + (i,), d, relative)
    #     s.faces = faces
    #     self.smap[s] = s
    #     return s
    # def update(self, points, relative_idx, sort=True):
    #     self.relative_vertex_idx = self.relative_vertex_idx.union({self.n_pts + i for i in relative_idx})
    #     RipsComplex.update(self, points, sort)


class RelativeScalarRips(ScalarRipsComplex, RelativeRipsComplex):
    def __init__(self, points, fun, relative_idx, eps, dim):
        # self.rips_idx = sorted(range(len(self)), key=lambda i: self[i].data)
        # self.rips_perm = {j : i for i,j in enumerate(self.rips_idx)}
        perm = sorted(range(len(fun)),key=lambda i: fun[i])
        imap = {j : i for i,j in enumerate(perm)}
        relative_idx = {imap[i] for i in relative_idx}
        points, self.fun = points[perm], [fun[i] for i in perm]
        RelativeRipsComplex.__init__(self, points, relative_idx, eps, dim, False)
        self.rips_idx, self.rips_perm = self.sort_perm('rips')
        self.rips_sub_idx = sorted(self.sub_idx, key=lambda i: self.rips_perm[i])
        # self.rips_sub_idx =
        self.rips_sub_imap = {j : i for i,j in enumerate(self.rips_sub_idx)}
        # self.rips_idx = sorted(range(len(self)), key=lambda i: self[i].data)
        # self.rips_perm = {j : i for i,j in enumerate(self.rips_idx)}
        # # self.rips_boundary = None
    def get_boundary(self, key, relative=True, sub=False):
        if key == 'rips':
            if relative:
                return [{self.rips_perm[t.index] for t in self[i].faces \
                                 if not t.relative} for i in self.rips_idx]
            elif sub:
                return [{self.rips_sub_imap[t.index] for t in self[i].faces} \
                                                for i in self.rips_sub_idx]
            return ScalarRipsComplex.get_boundary(self, key)
            # return [{self.rips_perm[t.index] for t in self[i].faces} for i in self.rips_idx]
        elif key == 'scalar':
            return RelativeRipsComplex.get_boundary(self, key, relative, sub)
        raise Exception('unknown value key %s' % key)
    def get_simplex(self, i, key, sub=False):
        if sub:
            if key == 'rips':
                return self[self.rips_sub_idx[i]]
            elif key == 'scalar':
                return self[self.sub_idx[i]]
        return ScalarRipsComplex.get_simplex(self, i, key)
    def make_vertex(self, i):
        relative = i in self.relative_vertex_idx
        return ScalarRipsComplex.make_vertex(self, i, relative)
    def make_simplex(self, i, t):
        relative = t.relative and self.smap[(i,)].relative
        return ScalarRipsComplex.make_simplex(self, i, t, relative)
    def sort_perm(self, key):
        idx = sorted(range(len(self)), key=lambda i: self[i].values[key])
        perm = {j : i for i,j in enumerate(idx)}
        return idx, perm
    def sort(self, key=None):
        return RelativeRipsComplex.sort(self, key)
        # idx, perm = self.sort_perm(key)
        # self.sub_idx = [perm[i] for i in sorted(self.sub_idx, key=lambda i: perm[i])]
        # self.sub_imap = {j : i for i,j in enumerate(self.sub_idx)}
        # self.filtration = [self[i].set_index(j) for j,i in enumerate(idx)]
    def chain_boundary(self, chain):
        bdy = set()
        for s in chain:
            if not any(v in self.relative_vertex_idx for v in s):
            # if not any(any(v in bidx for v in s) for bidx in self.border_idx):
                for t in s.faces:
                    if t in bdy:
                        bdy.remove(t)
                    else:
                        bdy.add(t)
        return bdy
    # def get_value(self, i, key):
    #     if key == 'rips':
    #         return self.get_simplex(i, key).values['rips']
    #     elif key == 'scalar':
    #         return self.get_simplex(i, key).values['scalar']
    #     raise Exception('unknown value key %s' % key)
    # def get_simplex(self, i, key):
    #     if key == 'rips':
    #         return self[self.rips_idx[i]]
    #     elif key == 'scalar':
    #         return self[i]
    #     raise Exception('unknown value key %s' % key)
    # def make_vertex(self, i):
    #     relative = i in self.relative_vertex_idx
    #     values = {'rips' : 0, 'scalar' : self.fun[i]}
    #     s = Simplex((i,), 0, relative, values)
    #     self.smap[s] = s
    #     return s
    # def make_simplex(self, i, t):
    #     faces = [self.smap[f] for f in t.star(i)] + [t]
    #     d = max(f.data for f in faces) if t.dim > 0 else self.dmat[i,t[0]]
    #     relative = t.relative and self.smap[(i,)].relative
    #     fv = max(t.values['scalar'], self.fun[i])
    #     values = {'rips' : d, 'scalar' : fv}
    #     s = Simplex(t + (i,), d, relative, values)
    #     s.faces = faces
    #     self.smap[s] = s
    #     return s
    # def update(self, points, fun, relative_idx):
    #     self.fun += fun
    #     RelativeRipsComplex.update(self, points, relative_idx, False)
    #     self.sort('scalar')
    #     self.rips_idx, self.rips_perm = self.sort_perm('rips')

class RelativeTorusScalarRips(RelativeScalarRips, TorusScalarRips):
    def __init__(self, points, fun, relative_idx, eps, dim):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(points)
        density = kde.score_samples(points)
        # density = max(density) - density
        init_perm = sorted(range(len(fun)), key=lambda i: density[i]) # fun[i])
        init_imap = {j : i for i, j in enumerate(init_perm)}
        init_points, init_fun = points[init_perm], [fun[i] for i in init_perm]
        init_relative_idx = {init_imap[i] for i in relative_idx}
        self.torus_points, self.torus_vmap, self.torus_idxs, self.torus_offs = self.make_torus(init_points, eps)

        self.fun = [init_fun[i] for i in self.torus_idxs]
        RelativeRipsComplex.__init__(self, self.torus_points, init_relative_idx, eps, dim, False)
        self.sort('scalar')
        # self.torus_points, self.points = self.points, init_points[init_perm]
        # self.torus_fun, self.fun = self.fun, [init_fun[i] for i in init_perm]
        # self.n_pts = len(self.points)
        self.build_faces()
        self.rips_idx, self.rips_perm = self.sort_perm('rips')
        self.rips_sub_idx = sorted(self.sub_idx, key=lambda i: self.rips_perm[i])
        self.rips_sub_imap = {j : i for i,j in enumerate(self.rips_sub_idx)}
    def make_vertex(self, i):
        j = self.torus_vmap[i]
        relative = j in self.relative_vertex_idx
        return TorusScalarRips.make_vertex(self, i, relative)
    def make_simplex(self, i, t, d):
        j = self.torus_vmap[i]
        relative = t.relative and self.smap[(j,)].relative
        return TorusScalarRips.make_simplex(self, i, t, d, relative)
    def get_lower_nbrs(self, i):
        return TorusRipsComplex.get_lower_nbrs(self,i)
    def add_cofaces(self, t, N):
        return TorusRipsComplex.add_cofaces(self, t, N)
    def add(self, s):
        return RelativeTorusRips.add(self, s)
    def tcc(self, dgm, connected_components=1):
        return RelativeRipsComplex.tcc(self, dgm, connected_components)
# class TorusScalarRips(RelativeScalarRips):
#     def __init__(self, points, fun, eps, dim):
#         self.init_points = points
#         self.relative_pts, relative_idx = self.make_torus(points, eps)
#         r_idx = range(len(self.relative_pts))
#         all_pts = np.vstack([self.relative_pts, points])
#         fun = [fun[i] for i in relative_idx] + fun
#         RelativeScalarRips.__init__(self, all_pts, fun, r_idx, eps, dim)
#     def make_torus(self, points, eps):
#         l = 2*np.pi
#         a, b = eps, l - eps
#         relative_pts, relative_idx = [], []
#         for i,p in enumerate(points):
#             if p[0] <= a and p[1] <= a:
#                 relative_pts.append(p + np.array([l,l]))
#                 relative_idx.append(i)
#             elif p[0] >= b and p[1] <= a:
#                 relative_pts.append(p + np.array([-l,l]))
#                 relative_idx.append(i)
#             elif p[0] <= a and p[1] >= b:
#                 relative_pts.append(p + np.array([l,-l]))
#                 relative_idx.append(i)
#             elif p[0] >= b and p[1] >= b:
#                 relative_pts.append(p + np.array([-l,-l]))
#                 relative_idx.append(i)
#             if p[0] <= a:
#                 relative_pts.append(p + np.array([l,0]))
#                 relative_idx.append(i)
#             if p[1] <= a:
#                 relative_pts.append(p + np.array([0,l]))
#                 relative_idx.append(i)
#             if p[0] >= b:
#                 relative_pts.append(p + np.array([-l,0]))
#                 relative_idx.append(i)
#             if p[1] >= b:
#                 relative_pts.append(p + np.array([0,-l]))
#                 relative_idx.append(i)
#             # relative_imap[len(relative_pts)] = i
#         relative_pts = np.vstack(relative_pts) if len(relative_pts) else []
#         return relative_pts, relative_idx
#     # def is_relative_chain(self, chain):
#     #     chain_boundary_pts = {i for e in self.chain_boundary(chain) for i in e}
#     #     return all(i in self.relative_vertex_idx for i in chain_boundary_pts)
#     # def make_simplex(self, i, d=0, t=()):
#     #     relative = i in self.relative_vertex_idx
#     #     if isinstance(t, Simplex):
#     #         relative = relative and t.relative
#     #         d = max(t.data, d)
#     #     return Simplex(t + (i,), d, relative)
#     # def update(self, points, fun):
#     #     relative_pts, relative_idx = self.make_torus(points, self.eps)
#     #     RelativeScalarRips.update(self, points, fun, relative_idx)

# class TorusScalarRips(TorusRipsComplex):
#     def __init__(self, points, fun, eps, dim, n_chunk=64):
#         perm = sorted(range(len(fun)),key=lambda i: fun[i])
#         points, fun = points[perm], [fun[i] for i in perm]
#         self.fun = fun
#         TorusRipsComplex.__init__(self, points, eps, dim, False)
#         self.rips_idx = sorted(range(len(self)), key=lambda i: self[i].data)
#         self.rips_perm = {j : i for i,j in enumerate(self.rips_idx)}
#         self.rips_boundary = None
#     def get_boundary(self, key):
#         if key == 'rips':
#             if self.rips_boundary is None:
#                 self.rips_boundary = [{self.rips_perm[j] for j in self.boundary[i]} for i in self.rips_idx]
#             return self.rips_boundary
#         elif key == 'scalar':
#             return self.boundary
#         raise Exception('unknown value key %s' % key)
#     def get_value(self, i, key):
#         if key == 'rips':
#             return self.get_simplex(i, key).values['rips']
#         elif key == 'scalar':
#             return self.get_simplex(i, key).values['scalar']
#         raise Exception('unknown value key %s' % key)
#     def get_simplex(self, i, key):
#         if key == 'rips':
#             return self[self.rips_idx[i]]
#         elif key == 'scalar':
#             return self[i]
#         raise Exception('unknown value key %s' % key)
#     def make_simplex(self, i, d=0, t=()):
#         it = self.torus_vmap[i]
#         fv = self.fun[it]
#         if isinstance(t, Simplex):
#             d = max(t.data, d)
#             fv = max(t.values['scalar'], fv)
#         values = {'rips' : d, 'scalar' : fv}
#         return BiSimplex(t + (it,), d, values)
#     def chain_boundary(self, chain):
#         bdy = set()
#         for s in chain:
#             if not any(any(v in bidx for v in s) for bidx in self.border_idx):
#                 for t in s.faces:
#                     if t in bdy:
#                         bdy.remove(t)
#                     else:
#                         bdy.add(t)
#         return bdy



# class ScalarRipsComplex(RipsComplex):
#     def __init__(self, values, data, eps, dim):
#         perm = sorted(range(len(values)), key=lambda i: values[i])
#         self.values = values[perm]
#         self.smap, self.boundary = {}, []
#         RipsComplex.__init__(self, data[perm], eps, dim, False)
#         # self.rips_filtration = [s.swap_value() for s in self.filtration]
#         # self.rips_filtration.sort()
#         self.rips_perm = sorted(range(len(self)), key=lambda i: self[i].value)
#         self.rips_imap = {j : i for i,j in enumerate(self.rips_perm)}
#         self.rips_filtration = [self[i].swap_value() for i in self.rips_perm]
#         self.rips_boundary = [{self.rips_imap[j] for j in self.boundary[i]} for i in self.rips_perm]
#     def make_simplex(self, v, d):
#         return BiSimplex(v, d, self.values[max(v)])
#     def add(self, s):
#         self.smap[s] = len(self.filtration)
#         bdy = {self.smap[t] for t in s.face_it()} if s.dim > 0 else set()
#         self.boundary.append(bdy)
#         FilteredSimplicialComplex.add(self, s.swap_value())
# class TorusRipsComplex(RipsComplex):
#     def __init__(self, points, eps, dim, sort=True, n_chunk=64):
#         FilteredSimplicialComplex.__init__(self, dim)
#         self.points = points
#         self.eps = eps
#         self.n_pts = len(self.points)
#         self.torus_points, self.torus_vmap, self.border_idx = self.make_torus()
#         self.dmat = distance_matrix(self.torus_points, self.torus_points)
#         self.lower_nbrs = {}
#         # for i in tqdm(range(len(self.torus_points))):
#         for i in range(len(self.torus_points)):
#             self.lower_nbrs[i] = self.get_lower_nbrs(i)
#             self.add_cofaces(self.make_simplex(i), self.lower_nbrs[i])
#         if sort:
#             self.sort()
#     # def get_points(self, chain, key=None):
#     #     l = 2*np.pi
#     #     I, _, K = self.border_idx
#     #     pts = []
#     #     for s in chain:
#     #         tiles = [[l if v in I else 0, l if v in K else 0] for v in s]
#     #         for i,j in combinations(s,2):
#     #             p, q = self.points[i], self.points[j]
#     #             if la.norm(p - q) > self.eps:
#     #                 oi = np.array([l if i in I else 0, l if i in K else 0])
#     #                 oj = np.array([l if j in I else 0, l if j in K else 0])
#     #                 pqs = np.array([[p+oi,q], [p,q+oj], [p+oi, q+oj]])
#     #
#     #             if a
#     #         if any(any(v in b for v in s) for b in self.border_idx):
#     #
#     #     return [self.points[s] for s in chain]
#     def add(self, s):
#         if not s in self.imap:
#             self.imap[s] = len(self.filtration)
#             self.filtration.append(s)
#             self.boundary.append(self.get_faces(s))
#             # self.smap[s] = s
#         elif s.data < self.filtration[self.imap[s]].data:
#             self.filtration[self.imap[s]] = s
#             # self.smap[s] = s
#         SimplicialComplex.add(self, s)
#     def make_simplex(self, i, d=0, t=()):
#         d = max(t.data, d) if isinstance(t, Simplex) else d
#         return Simplex(t + (self.torus.vmap[i],), d)
#     def make_torus(self):
#         l = 2*np.pi
#         tiles = np.array([[l,0],[l,l],[0,l]])
#         J = [i for i,p in enumerate(self.points) if p[0] <= self.eps and p[1] <= self.eps]
#         I = [i for i,p in enumerate(self.points) if p[0] <= self.eps]# and p[1] > self.eps]
#         K = [i for i,p in enumerate(self.points) if p[1] <= self.eps]# and p[0] > self.eps]
#         border_idx = [I, J, K]
#         vmap = {j : i for j,i in enumerate(list(range(self.n_pts)) + I + J + K)}
#         torus_points = np.vstack([self.points,
#                                 self.points[I] + tiles[0],
#                                 self.points[J] + tiles[1],
#                                 self.points[K] + tiles[2]])
#         return torus_points, vmap, border_idx
