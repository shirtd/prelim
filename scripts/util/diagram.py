import numpy as np
import numpy.linalg as la
from numpy.random import rand
from itertools import combinations
from functools import reduce
from matplotlib import cm
from tqdm import tqdm
import time
import dionysus as dio

def chain_boundary(chain):
    bdy = set()
    for s in chain:
        for t in s.faces:
            if t in bdy:
                bdy.remove(t)
            else:
                bdy.add(t)
    return bdy

class Simplex(tuple):
    def __new__(cls, vertices, data, index, dindex, faces, relative=False):
        return tuple.__new__(cls, tuple(vertices))
    def __init__(self, vertices, data, index, dindex, faces, relative=False):
        tuple.__init__(self)
        self.dim = len(self) - 1
        self.data, self.index, self.dindex = data, index, dindex
        self.faces, self.relative = faces, relative
        self.minv = data if self.dim == 0 else min(t.minv for t in self.faces)
    def closure(self):
        if self.dim < 1:
            return set()
        sub = set(self.faces).union({t for s in self.faces for t in s.closure()})
        return {self}.union(sub)
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
        return '%s:%0.4f' % (tuple.__repr__(self), self.data)

class DioFilt:
    def __init__(self, filt, field, omega=-np.inf, relative=True, rev=False, super=False):
        self.fv = field.T.flatten()
        self.omega, self.relative, self.rev = omega, relative, rev
        self.smap, self.filt = {}, []
        for i,s in tqdm(list(enumerate(filt))):
            minv, maxv = min(self.fv[v] for v in s),  max(self.fv[v] for v in s)
            if (not relative # and ((rev and s.data > omega) or (not rev and s.minv < omega))):
                and ((rev and (maxv if super else s.data) > omega)
                    or (not rev and (s.data if super else minv) < omega))):
                continue
            faces = [self.smap[tuple(t)] for t in s.boundary()]
            rel = (relative and ((not rev and (maxv if super else s.data)  < omega)
                                or (rev and (s.data if super else minv) > omega)))
            ss = Simplex(s, s.data, len(self.filt), i, faces, rel)
            self.filt.append(ss)
            self.smap[ss] = ss
    def __getitem__(self, i):
        return self.filt[i]
    def __len__(self):
        return len(self.filt)
    def __iter__(self):
        for s in self.filt:
            yield s
    def get_value(self, i):
        return self[i].data
    def get_simplex(self, i):
        return self[i]
    def get_boundary(self):
        return [{t.index for t in s.faces if not t.relative} for s in self]
    def dio_close(self, L):
        cl = {t for s in L for t in s.closure()}
        Fcl = set(self).union(cl)
        F = dio.Filtration([dio.Simplex(s, s.data) for s in clres])
        F.sort()
        return F


class DiagramPoint:
    def __init__(self, F, c, b, d=None):
        self.chain = c
        self.is_inf = d is None
        self.birth = F.get_value(b)
        self.death = F.get_value(d) if not self.is_inf else np.inf
        self.birth_index, self.death_index = b, d
        self.dim = F[b].dim
    def __eq__(self, other):
        return (self.birth_index == other.birth_index
            and self.death_index == other.death_index)
    def as_np(self, lim):
        return [self.birth, 1.2*lim if self.is_inf else self.death]
    def __repr__(self):
        return '[%0.4f\t%0.4f]' % (self.birth, self.death)
    def __iter__(self):
        for i in self.chain:
            yield i

class Diagram:
    def __init__(self, F):
        self.F = F
        self.relative = F.relative
        self.lim = -np.inf
        self.D = self.F.get_boundary()
        self.V = [{i} for i in range(len(self.D))]
        self.diagram = {}
        self.diagonal = {}
        self.extra = []
    def get_simplex(self, i):
        return self.F[i]
    def get_value(self, i):
        return self.F[i].data
    def get_inf(self, dim):
        return [pt for pt in self[dim] if pt.is_inf]
    def get_dead(self, dim, low=-np.inf, hi=np.inf, diag=False):
        it = self[dim] + self.diagonal[dim] if diag else self[dim]
        return [pt for pt in it if low < pt.death and pt.death < hi]
    def get_born(self, dim, low=-np.inf, hi=np.inf, diag=False):
        it = self[dim] + self.diagonal[dim] if diag else self[dim]
        return [pt for pt in it if low < pt.birth and pt.birth < hi]
    def make_point(self, c, low, i=None):
        return DiagramPoint(self.F, c, low, i)
    def add_pair(self, low, i, c):
        pt = self.make_point(c, low, i)
        if pt.death > self.lim:
            self.lim = pt.death
        if self.get_value(low) == self.get_value(i):
            if not pt.dim in self.diagonal:
                self.diagonal[pt.dim] = []
            self.diagonal[pt.dim].append(pt)
            return pt
        if not pt.dim in self.diagram:
            self.diagram[pt.dim] = []
        self.diagram[pt.dim].append(pt)
        return pt
    def add_unpair(self, i, c):
        pt = self.make_point(c, i)
        if self.get_simplex(i).dim == 3:
            self.extra.append(pt)
            return pt
        if not pt.dim in self.diagram:
            self.diagram[pt.dim] = []
        self.diagram[pt.dim].append(pt)
        if pt.birth > self.lim:
            self.lim = pt.birth
        return pt
    def as_list(self, dim=None):
        if dim is None:
            return {d : self.as_list(d) for d in self.diagram}
        if not dim in self.diagram:
            return np.ndarray((0,2), dtype=float)
        return [[p.birth, p.death] for p in self[dim]]
    def as_np(self, dim=None, lim=None):
        lim = self.lim if lim is None else lim
        if dim is None:
            return {d : self.as_np(d, lim) for d in self.diagram}
        if not dim in self.diagram:
            return np.ndarray((0,2), dtype=float)
        return np.array([p.as_np(lim) for p in self[dim]])
    def __getitem__(self, dim):
        return self.diagram[dim] if dim in self.diagram else []
    def __iter__(self):
        for d in sorted(self.diagram):
            yield self.diagram[d]
    def get_chain(self, pt):
        return [self.get_simplex(i) for i in pt.chain]
    # def get_chain_points(self, pt):
    #     return self.F.points[list({v for s in self.get_chain(pt) for v in s})]
    def betti(self, dim):
        return len(self.get_inf(dim))

def phcol(F):
    dgm = Diagram(F)
    pairs, unpairs, diag = {}, {}, {}
    for i in tqdm(range(len(dgm.D))):
        if dgm.D[i]:
            low = max(dgm.D[i])
            while low in pairs:
                dgm.D[i] ^= dgm.D[pairs[low]]
                dgm.V[i] ^= dgm.V[pairs[low]]
                if dgm.D[i]:
                    low = max(dgm.D[i])
                else:
                    low = None
                    unpairs[i] = dgm.V[i]
            if low is not None:
                pairs[low] = i
                dgm.D[low].clear()
                dgm.add_pair(low, i, dgm.D[i])
                if low in unpairs:
                    del unpairs[low]
        elif not (F.relative and F[i].relative):
            unpairs[i] = dgm.V[i]
    for k, v in unpairs.items():
        dgm.add_unpair(k, v)
    return dgm

# class FiltWrap:
#     def __init__(self, dat, dim=1, thresh=0, cut=-np.inf, relative=True):
#         self.filt = DioFilt(dat['filt'], cut, relative)
#         self.dim = dim
#         self.dgm = [pt for pt in dat['dgms'][dim] if pt.birth >= cut and pt.death - pt.birth > thresh]
#         self.pairs = [(pt,)+self.get_pair(pt) for pt in self if self.is_paired(pt)]
#         self.dgm_np = np.array([[p.birth, p.death] for p in self])
#         self.bmap = {tuple(b) : (p, b, d) for p, b, d in self.pairs}
#         self.dmap = {tuple(d) : (p, b, d) for p, b, d in self.pairs}
#         self.bmap_c = {b : self.get_chain(p) for b, (p,_,_) in self.bmap.items()}
#         self.dmap_c = {d : self.get_chain(p) for d, (p,_,_) in self.dmap.items()}
#         self.prio = None
#     def __iter__(self):
#         for p in self.dgm:
#             yield p
#     def set_prio(self, prio):
#         if len(prio):
#             self.prio = prio
#         else:
#             self.prio = None
#     def remove_prio(self):
#         self.prio = None
#     def pair(self, p):
#         return self.hom.pair(p.data)
#     def is_paired(self, p):
#         return self.pair(p) != self.hom.unpaired
#     def get_birth(self, pt):
#         return self.filt[pt.data]
#     def get_death(self, pt):
#         return self.filt[self.hom.pair(pt.data)]
#     def get_chain(self, pt):
#         return self.hom[self.hom.pair(pt.data)]
#     def get_chain_idx(self, pt):
#         chain = self.get_chain(pt)
#         return [list(self.filt[c.index]) for c in chain]
#     def get_pair(self, pt):
#         return (self.get_birth(pt), self.get_death(pt))
#     def match_death(self, t):
#         return [(pbd,t.dmap[d]) for d,pbd in self.dmap.items() if d in t.dmap and t.dmap[d][1] != pbd[1]]
#     def unmatched_death(self, t):
#         return [pbd for d, pbd in self.dmap.items() if not d in t.dmap]
#     def closest(self, q):
#         dgm = self.dgm if self.prio is None else self.prio
#         return min(dgm, key=lambda p: la.norm(q - np.array([p.birth,p.death])))
