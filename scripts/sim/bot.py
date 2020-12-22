import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.random import rand
from itertools import combinations
from scipy.spatial import distance_matrix
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib import cm
from tqdm import tqdm
import pickle as pkl
import os, sys

def get_pt(t, l):
    return np.array([l * np.cos(t), l * np.sin(t)])

def get_pos(t0=np.pi/2, t1=np.pi, l0=1, l1=1):
    p = get_pt(t0, l0)
    q = get_pt(t0 - np.pi + t1, l1) + p
    return np.vstack([p, q])

def point_line_distance(a, b, v):
    ab, av, bv = b - a, v - a, v - b
    if av.dot(ab) <= 0.0: return la.norm(av)
    if bv.dot(ab) >= 0.0: return la.norm(bv)
    return la.norm(np.cross(ab, av))/ la.norm(ab)

def point_curve_distance(c, p):
    return min(point_line_distance(u, v, p) for u, v in zip(c[:-1], c[1:]))

class RobotArm:
    def __init__(self, t0, t1, l0=1, l1=1):
        self.t0, self.t1, self.l0, self.l1 = t0, t1, l0, l1
        self.p = get_pt(self.t0, self.l0)
        self.q = get_pt(self.t0 - np.pi + self.t1, self.l1) + self.p
        self.arm = np.vstack([[0, 0], self.p, self.q])
    def __len__(self):
        return self.l0 + self.l1
    def __getitem__(self, i):
        return self.arm[i]
    def dist(self, q, mode='head'):
        if mode == 'head':
            return la.norm(q - self.arm[2])
        elif mode == 'body':
            return point_curve_distance(self.arm, q)
    def plot(self, axis, clear=True):
        if clear:
            axis.cla()
            axis.set_xlim(-len(self), len(self))
            axis.set_ylim(-len(self), len(self))
        return [l for l in axis.plot(self[:,0], self[:,1], marker='o')]

def animate_seq(self, axis, seq):
    for t0, t1 in seq:
        arm = RobotArm(t0, t1, self.l0, self.l1)
        arm.plot(axis, clear=True)
        for src in self.src:
            axis.scatter(src[0], src[1], c='red', s=30*w, zorder=0)
        plt.pause(1)

class Configuration:
    def __init__(self, n=1000, ls=[1.,1.],
                omega=None, eps=None, nsrc=1,
                fun='sum', mode='head'):
        self.n = 0
        self.l0, self.l1 = ls
        self.src = [(self.l0+self.l1) * (2*rand(2) - 1) for _ in range(nsrc)]
        self.config, self.dist = None, []
        self.eps, self.omega = eps, omega
        self.fun, self.mode = fun, mode
        self.max_dist = -np.inf
        self.run_n(n)
    def __getitem__(self, i):
        return self.config[i]
    def __len__(self):
        return len(self.config)
    def __iter__(self):
        for t in self.config:
            yield t
    def run_n(self, n):
        for _ in range(n):
            self.run_random()
    def get_lipschitz(self):
        it = combinations(range(len(self)),2)
        return max(abs(self.dist[i] - self.dist[j]) \
                        / la.norm(self[i] - self[j]) \
                        for i, j in tqdm(it))
    def run_random(self, t=None):
        t = 2*np.pi*rand(2) if t is None else t
        arm = RobotArm(t[0], t[1], self.l0, self.l1)
        if self.fun == 'sum':
            # d = sum(arm.dist(s, self.mode) ** 2 / (i+1) for i,s in enumerate(self.src)) / len(self.src) # ** 2
            # d = sum(arm.dist(s, self.mode) / (i+1) for i,s in enumerate(self.src)) / len(self.src) # ** 2
            d = sum(arm.dist(s, self.mode) for i,s in enumerate(self.src)) / len(self.src) # ** 2
        elif self.fun == 'min':
            d = min(arm.dist(s, self.mode) for i,s in enumerate(self.src))
        if self.config is None:
            self.config = t
        else:
            self.config = np.vstack([self.config, t])
        self.dist.append(d)
        self.max_dist = d if d > self.max_dist else self.max_dist
        self.n += 1
    def get_rips(self):
        rips = RelativeRipsScalar(self)
        return rips.F, rips.G, rips.S, rips.T
    def plot_config(self, axis, relative=False, full=False, clear=False, show=False):
        if clear: axis.cla()
        l = 2*np.pi
        axis.set_xlim(-3*np.pi/2, l + 3*np.pi/2)
        axis.set_ylim(-3*np.pi/2, l + 3*np.pi/2)
        axis.plot([0, l, l, 0, 0], [0, 0, l, l, 0], c='black', zorder=0)
        if relative:
            cmap = cm.get_cmap('viridis_r')
            nrm = lambda d: (d - min(self.dist)) / (self.omega - min(self.dist))
            C = [[0, 0, 0, 0] if d > self.omega else cmap(int(30 + 226*nrm(d)**2)) for d in self.dist]
        elif full:
            cmap = cm.get_cmap('viridis_r')
            nrm = lambda d: (d - min(self.dist)) / (max(self.dist) - min(self.dist))
            C = [cmap(int(30 + 226*nrm(d)**2)) for d in self.dist]
        else:
            cmap = cm.get_cmap('viridis_r')
            nrm = lambda d: (d - self.omega) / (max(self.dist) - self.omega)
            C = [[1, 0, 0, 0.3] if d <= self.omega else cmap(int(30 + 226*nrm(d)**2)) for d in self.dist]
        tiles = [[0,0],[l,0],[-l,0],[0,l],[0,-l],[l,l],[-l,-l],[l,-l],[-l,l]]
        for x,y in tiles:
            axis.scatter(self[:,0]+x, self[:,1]+y, c=C, s=5)
        patches = []
        for p in self:
            patches.append(Circle(p, self.eps / 2))
        coll = PatchCollection(patches, alpha=0.1, zorder=0)
        axis.add_collection(coll)
        if show:
            plt.pause(0.1)
    def save_plot(self, name, dir='figures'):
        # plt.pause(0.1)
        try:
            os.mkdir(dir)
        except:
            pass
        i = 0
        fname = os.path.join(dir, '%s-%d.png' % (name, i))
        while os.path.exists(fname):
            i += 1
            fname = os.path.join(dir, '%s-%d.png' % (name, i))
        print('\t* saving %s' % fname)
        plt.savefig(fname, dpi=200)
    def pickle(self, name, dir='arm_data'):
        try:
            os.mkdir(dir)
        except:
            pass
        i = 0
        fname = os.path.join(dir, '%s-%d.pkl' % (name, i))
        while os.path.exists(fname):
            i += 1
            fname = os.path.join(dir, '%s-%d.pkl' % (name, i))
        with open(fname, 'wb') as f:
            print('\t* saving %s' % fname)
            pkl.dump(self, f)
