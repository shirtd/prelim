import numpy as np
import numpy.linalg as la
from numpy.random import rand
from itertools import combinations
from matplotlib import cm
from tqdm import tqdm
import time


class DiagramPoint:
    def __init__(self, F, c, b, d=None, key=None, sub=False):
        self.chain = c
        self.is_inf = d is None
        self.birth = F.get_value(b, key, sub)
        self.death = F.get_value(d, key, sub) if not self.is_inf else np.inf
        self.birth_index, self.death_index = b, d
        self.dim = F.get_simplex(b, key, sub).dim
        self.key = key
    def __eq__(self, other):
        return (self.birth_index == other.birth_index
            and self.death_index == other.death_index)
    def as_np(self, lim):
        return [self.birth, 2*lim if self.is_inf else self.death]
    def __repr__(self):
        return '[%0.4f\t%0.4f]' % (self.birth, self.death)
    def __iter__(self):
        for i in self.chain:
            yield i

class Diagram:
    def __init__(self, F, key=None, relative=False, sub=False, coh=False):
        self.F = F
        self.key = key
        self.relative = relative
        self.sub = sub
        self.coh = coh
        self.lim = -np.inf
        self.D = self.F.get_boundary(key, relative, sub)
        self.V = [{i} for i in range(len(self.D))]
        self.diagram = {}
        self.diagonal = {}
        self.extra = []
    def get_simplex(self, i):
        return self.F.get_simplex(i, self.key, self.sub)
    def get_value(self, i):
        return self.F.get_value(i, self.key, self.sub)
    def get_inf(self, dim):
        return [pt for pt in self[dim] if pt.is_inf]
    def get_dead(self, dim, low=-np.inf, hi=np.inf, diag=False):
        it = self[dim] + self.diagonal[dim] if diag else self[dim]
        return [pt for pt in it if low < pt.death and pt.death < hi]
    def get_born(self, dim, low=-np.inf, hi=np.inf, diag=False):
        it = self[dim] + self.diagonal[dim] if diag else self[dim]
        return [pt for pt in it if low < pt.birth and pt.birth < hi]
    def make_point(self, c, low, i=None):
        return DiagramPoint(self.F, c, low, i, self.key, self.sub)
    def add_pair(self, low, i, c):
        if self.coh:
            n = len(self.D)
            low, i = n - i - 1, n - low - 1
            c = {n - j - 1 for j in c}
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
        if self.coh:
            n = len(self.D)
            i = n - i - 1
            c = {n - j - 1 for j in c}
        pt = self.make_point(c, i) # DiagramPoint(self.F, c, i)
        if self.get_simplex(i).dim == 3:
            self.extra.append(pt)
            return pt
        if not pt.dim in self.diagram:
            self.diagram[pt.dim] = []
        self.diagram[pt.dim].append(pt)
        if pt.birth > self.lim:
            self.lim = pt.birth
        return pt
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
    def get_chain_points(self, pt):
        return self.F.points[list({v for s in self.get_chain(pt) for v in s})]
    def betti(self, dim):
        return len(self.get_inf(dim))

def transpose_boundary(B, rev=False):
    n = len(B)
    F = [set() for i in range(n)]
    fi = lambda i: n - i - 1 if rev else i
    for j, s in enumerate(B):
        for i in s:
            F[fi(i)].add(fi(j))
    return F

def phcol(scalar_rips, relative=False, key=None, sub=False, coh=False):
    dgm = Diagram(scalar_rips, key, relative, sub, coh)
    # D = scalar_rips.get_boundary(key, relative, sub)
    # n = len(D)
    # fi = lambda i: n - i - 1 if coh else i
    # if len(relative_idx):
    #     relative_idx = {scalar_rips.rips_imap[i] for i in relative_idx}
    # else:
    #     dgm = Diagram(scalar_rips.filtration, coh)
    #     D = scalar_rips.boundary.copy()
    # if coh:
    #     D = transpose_boundary(D, True)
    # V = [{i} for i in range(n)]
    pairs, unpairs, diag = {}, {}, {}
    for i in range(len(dgm.D)):
        # if scalar_rips.is_relative(i, key):
        #     continue
        if dgm.D[i]:
            low = max(dgm.D[i])
            while low in pairs:
                dgm.D[i] ^= dgm.D[pairs[low]]
                dgm.V[i] ^= dgm.V[pairs[low]]
                if dgm.D[i]:
                    low = max(dgm.D[i])
                else:
                    low = None
                    # if dgm.get_simplex(i).dim < 3:
                    unpairs[i] = dgm.V[i]
                    # else:
                    #     dgm.extra.append(dgm.make_point(c, i))
                    # break
            if low is not None:
                pairs[low] = i
                dgm.D[low].clear()
                dgm.add_pair(low, i, dgm.V[i] if coh else dgm.D[i])
                if low in unpairs:
                    del unpairs[low]
        elif not (relative and scalar_rips.is_relative(i, key)):
            unpairs[i] = dgm.V[i]
    for k, v in unpairs.items():
        dgm.add_unpair(k, v)
    # dgm.D = D
    # dgm.V = V
    return dgm

# def phcol(scalar_rips, relative=False, key=None, coh=False):
#     n = len(scalar_rips)
#     fi = lambda i: n - i - 1 if coh else i
#     D = scalar_rips.get_boundary(key)
#     dgm = Diagram(scalar_rips, key, coh)
#     # if rips and len(relative_idx):
#     #     dgm = Diagram(scalar_rips.rips_filtration, coh)
#     #     D = scalar_rips.rips_boundary.copy()
#     #     if len(relative_idx):
#     #         relative_idx = {scalar_rips.rips_imap[i] for i in relative_idx}
#     # else:
#     #     dgm = Diagram(scalar_rips.filtration, coh)
#     #     D = scalar_rips.boundary.copy()
#     if coh:
#         D = transpose_boundary(D, True)
#     V = [{i} for i in range(n)]
#     def lowest(S):
#         if relative:
#             S = list(filter(lambda i: not scalar_rips.is_relative(fi(i), key), S))
#         return max(S) if S else None
#     pairs, unpairs = {}, {}
#     for i in range(len(D)):
#         if relative and scalar_rips.is_relative(fi(i), key):
#             continue
#         low = lowest(D[i])
#         while low in pairs:
#             D[i] = D[i] ^ D[pairs[low]]
#             V[i] = V[i] ^ V[pairs[low]]
#             low = lowest(D[i])
#         if low is not None:
#             pairs[low] = i
#             dgm.add_pair(low, i, V[i] if coh else D[i])
#             if low in unpairs:
#                 del unpairs[low]
#         elif not i in pairs:
#             unpairs[i] = V[i]
#     for k, v in unpairs.items():
#         if scalar_rips.get_simplex(k, key).dim < 3:
#             dgm.add_unpair(k, v)
#     return dgm
