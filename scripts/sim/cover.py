from sim.bot import *
from sim.persist import *
from sim.simplicial import *
from sim.util import *

class IterCover(Configuration):
    def __init__(self, lips, seed=None, step=100, o_coef=2, e_coef=0.9,
                    n_init=500, ls=[1.,1.], omega=0, eps=1, nsrc=1,
                    fun='sum', mode='head'):
        print('\t\tseed:\t%d' % seed)
        print('  lipschitz constant:\t%0.4f' % lips)
        np.random.seed(seed)
        self.seed = seed
        self.lips, self.step, self.o_coef, self.e_coef = lips, step, o_coef, e_coef
        self.relative_idx, self.dgm, self.rel_dgm, self.sub_dgm = {}, {}, {}, {}
        self.connected_components = 1
        Configuration.__init__(self, n_init, ls, omega, eps, nsrc, fun, mode)
        self.added_pts = []
    def update_relative(self):
        self.relative_idx = {i for i,d in enumerate(self.dist) if d <= self.omega}
    def run_n(self, n=None):
        n = self.step if n is None else n
        Configuration.run_n(self, n)
        print('\tmax dist: %0.4f' % self.max_dist)
        if self.has_omega():
            self.update_relative()
        self.run_rips()
    def has_omega(self):
        return 0 < self.omega and self.omega < self.max_dist
    def run_rips(self):
        print('\t[ build rips')
        self.scalar_rips = RelativeTorusScalarRips(self.config, self.dist,
                                                self.relative_idx, self.eps, 3)
        self.run_persist('rips')
    def run_persist(self, key):
        print('\t | %s persist' % key)
        self.dgm[key] = phcol(self.scalar_rips, False, key)
        if self.has_omega():
            # print('\t | relative %s persist' % key)
            self.rel_dgm[key] = phcol(self.scalar_rips, True, key)
            self.sub_dgm[key] = phcol(self.scalar_rips, False, key, True)
    def is_covered(self):
        return self.dgm['rips'].betti(2) == self.connected_components
    def add_circumcenter(self, vs, new_pts, frac=1):
        if len(vs) == 3:
            new_pt = circumcenter(vs)
        else:
            new_pt = sum(vs) / len(vs)
        if new_pt is not None:
            new_pt[0] = new_pt[0] - 2*np.pi if new_pt[0] > 2*np.pi \
                        else new_pt[0] + 2*np.pi if new_pt[0] < 0 \
                        else new_pt[0]
            new_pt[1] = new_pt[1] - 2*np.pi if new_pt[1] > 2*np.pi \
                        else new_pt[1] + 2*np.pi if new_pt[1] < 0 \
                        else new_pt[1]
            rem = {i for i,p in enumerate(new_pts) if \
                la.norm(p-new_pt) < self.e_coef * self.eps / frac}
            if rem:
                new_pt = (new_pt + sum(new_pts[i] for i in rem)) / (len(rem)+1)
                new_pts = [p for i,p in enumerate(new_pts) if not i in rem]
            new_pts.append(new_pt)
        else:
            print('zero determinant', vs)
    def add_circumcenters(self, key='rips', frac=2, relative=False):
        new_pts = []
        dgm = self.rel_dgm[key] if relative and self.has_omega() else self.dgm[key]
        # for pt in dgm.get_dead(0, self.eps/frac):
        #     s = dgm.get_simplex(pt.death_index)
        #     vs = self.scalar_rips.points[list(s)] + s.offset
        #     self.add_circumcenter(vs, new_pts, frac)
        for pt in dgm.get_dead(1, self.eps/frac):#, diag=True):
            s = dgm.get_simplex(pt.death_index)
            vs = self.scalar_rips.points[list(s)] + s.offset
            self.add_circumcenter(vs, new_pts, frac)
        for pt in dgm[2]:
            s = dgm.get_simplex(pt.birth_index)
            vs = self.scalar_rips.points[list(s)] + s.offset
            self.add_circumcenter(vs, new_pts, frac)
        print('\t + adding %d circumcenters' % len(new_pts))
        for p in new_pts:
            if all(la.norm(p - q) > self.eps/10 for q in self.added_pts):
                self.run_random(p)
                self.added_pts.append(p)
        # return new_pts
    def is_omega(self, birth):
        return self.max_dist > birth + self.lips * self.eps / self.o_coef
    def set_omega(self, birth):
        if self.is_omega(birth):
            self.omega = birth + self.lips * self.eps / self.o_coef
            self.update_relative()
    def can_skip(self, birth):
        # eps = birth + self.lips * self.eps / self.o_coef
        eps = self.o_coef * (self.max_dist - birth) / self.lips
        return eps < self.eps * self.e_coef ** 2
        # return (birth < self.max_dist and self.max_dist < eps
        #         and eps < self.eps * self.e_coef ** 2)
            # and self.max_dist < birth + self.lips * self.eps / self.o_coef)
    def update_eps(self, birth, skip=False):
        if skip and self.can_skip(birth):
            self.eps = self.o_coef * (self.max_dist - birth) / self.lips
        else:
            self.eps *= self.e_coef
        self.add_circumcenters()
        self.set_omega(birth)
        self.run_rips()
    def plot_dgm(self, key, axs, show=False):
        lim = self.max_dist if key == 'scalar' else self.eps
        omega = self.omega if key == 'scalar' and self.has_omega() else None
        if self.has_omega():
            plot_dgm(axs[0], self.sub_dgm[key], lim, omega, clear=True)
            plot_dgm(axs[2], self.rel_dgm[key], lim, omega, clear=True)
        plot_dgm(axs[1], self.dgm[key], lim, omega, clear=True, show=show)
    def plot_reps(self, key, axs, colors=['blue', 'red'], show=False, torus=True):
        # plot_1_chain(axs[0], self.dgm[key], self.dgm[key].get_inf(1)[0], colors[0], 6, False)
        for pt in self.dgm[key].get_inf(1):
            plot_1_chain(axs[1], self.dgm[key], pt, colors[0], 6, False, torus)
        if self.has_omega():
            for pt in self.sub_dgm[key].get_inf(1):
                plot_1_chain(axs[0], self.dgm[key], pt, colors[0], 6, False, torus)
            for pt in self.rel_dgm[key].get_inf(1):
                plot_1_chain(axs[2], self.rel_dgm[key], pt, colors[0], 6, False, torus)
            for pt in self.rel_dgm[key].get_inf(2):
                plot_2_boundary(axs[2], self.rel_dgm[key], pt, colors[1], 6, False, torus)
        if show:
            plt.pause(0.1)
    def plot_config(self, axs, show=False):
        if self.has_omega():
            Configuration.plot_config(self, axs[0], relative=True, clear=True, show=False)
            Configuration.plot_config(self, axs[2], clear=True, show=False)
        Configuration.plot_config(self, axs[1], full=True, clear=True, show=show)
    def get_name(self):
        return '%s_%s_%d-%d' % (self.fun, self.mode, len(self.src), self.seed)
    def save_plot(self, name=None, **kw):
        name = self.get_name() if name is None else name
        Configuration.save_plot(self, name, **kw)
    def pickle(self, name=None, **kw):
        name = self.get_name() if name is None else name
        Configuration.pickle(self, name, **kw)
