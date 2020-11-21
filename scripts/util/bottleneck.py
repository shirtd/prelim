import dionysus as dio
from multiprocessing import Pool
from functools import partial
import numpy as np
import numpy.linalg as la
import pickle as pkl
from tqdm import tqdm
import os, sys
from util.plot import plot_diagram

def grid_coord(v, N):
        return [v//N, v%N]

def remove_inf(d):
    return dio.Diagram([(p, q) for p,q in d if q < np.inf])

def remove_infs(dgms):
    return [remove_inf(d) if len(d) else dio.Diagram([]) for d in dgms]

def do_bottlenecks(A, B):
    return [dio.bottleneck_distance(a, b) for a,b in zip(A, B)]

def pmap(fun, x, max_cores=None, *args, **kw):
    pool = Pool(max_cores)
    f = partial(fun, *args, **kw)
    try:
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

class Barcode:
    def __init__(self, G, cuts):
        self.N, _ = G.shape
        assert G.shape[1] == self.N
        self.G, self.cuts = G, cuts
        self.dgms, self.dgms_res, self.dgms_rel = self.get_barcodes()
        self.dgms_cut = [[[(p,q) for p,q in d if p >= c] for d in self.dgms] for c in self.cuts]
    def get_filts(self):
        filt = dio.fill_freudenthal(self.G.astype(float))
        rel_s = [[] for _ in self.cuts]
        res_s = [[] for _ in self.cuts]
        for s in filt:
            for i, l in enumerate(self.cuts):
                if s.data < l: # all(self.get_val(v) < l for v in s):
                    rel_s[i].append(s)
                elif all(self.get_val(v) >= l for v in s):
                    res_s[i].append(s)
        res_f = [dio.Filtration(s) for s in res_s]
        rel_f = [dio.Filtration(s) for s in rel_s]
        return filt, res_f, rel_f
    def get_dio_dat(self):
        filt, res_f, rel_f = self.get_filts()
        hom_res, hom_rel = [], []
        dgms_res, dgms_rel = [],[]
        for sf, lf in zip(res_f, rel_f):
            res_h = dio.homology_persistence(sf)
            rel_h = dio.homology_persistence(filt, relative=lf)
            dgms_res.append(dio.init_diagrams(res_h, sf))
            dgms_rel.append(dio.init_diagrams(rel_h, filt))
            hom_res.append(res_h)
            hom_rel.append(rel_h)
        hom = dio.homology_persistence(filt)
        dgms = dio.init_diagrams(hom, filt)
        return {'full' : {'filt' : filt, 'hom' : hom, 'dgms' : dgms},
                'res' : [{'filt' : f, 'hom' : h, 'dgms' : d} for f,h,d in zip(res_f, hom_res, dgms_res)],
                'rel' : [{'filt' : filt, 'hom' : h, 'dgms' : d} for h,d in zip(hom_rel, dgms_rel)]}
    def get_dio_wrap(self, dim, thresh):
        dat = self.get_dio_dat()
        return {'full' : DioWrap(dat['full'], dim, thresh),
                # 'cuts' : [DioWrap(dat['full'], dim, thresh, cut) for cut in self.cuts],
                'cuts' : [DioWrap(dat['full'], dim, cut=cut) for cut in self.cuts],
                'res' : [DioWrap(d, dim, thresh) for d in dat['res']],
                # 'res' : [DioWrap(d, dim) for d in dat['res']],
                'rel' : [DioWrap(d, dim, thresh) for d in dat['rel']]}
    def get_barcodes(self):
        dio_dat = self.get_dio_dat()
        dgms = self.dgms_list(dio_dat['full']['dgms'])
        dgms_res = [self.dgms_list(c['dgms']) for c in dio_dat['res']]
        dgms_rel = [self.dgms_list(c['dgms']) for c in dio_dat['rel']]
        return dgms, dgms_res, dgms_rel
    def dgm_list(self, dgm):
        return [(p.birth, p.death) for p in dgm] if len(dgm) else []
    def dgms_list(self, dgms):
        dgms = dgms + [[] for _ in range(len(dgms)-1,2)]
        return [self.dgm_list(d) for d in dgms]
    def grid_coord(self, v):
        return grid_coord(v, self.N)
    def get_val(self, v):
        i, j= self.grid_coord(v)
        return self.G[i, j]
    def query(self, dat, b, d, prio=None):
        pt = dat.closest(np.array([b, d]))
        b_idx, d_idx = pt.data, dat.pair(pt)
        c_idx = [c.index for  c in dat.hom[dat.pair(pt)]]
        return {'pt' : pt,
                'birth' : (b_idx, [self.grid_coord(v) for v in dat.filt[b_idx]]),
                'death' : (d_idx, [self.grid_coord(v) for v in dat.filt[d_idx]]),
                'chain' : (c_idx, [[self.grid_coord(v) for v in dat.filt[c]] for c in c_idx])}

class DioWrap:
    def __init__(self, dat, dim=1, thresh=0, cut=-np.inf):
        self.filt, self.hom, self.dim = dat['filt'], dat['hom'], dim
        self.dgm = [pt for pt in dat['dgms'][dim] if pt.birth >= cut and pt.death - pt.birth > thresh]
        self.pairs = [(pt,)+self.get_pair(pt) for pt in self if self.is_paired(pt)]
        self.dgm_np = np.array([[p.birth, p.death] for p in self])
        self.bmap = {tuple(b) : (p, b, d) for p, b, d in self.pairs}
        self.dmap = {tuple(d) : (p, b, d) for p, b, d in self.pairs}
        self.bmap_c = {b : self.get_chain(p) for b, (p,_,_) in self.bmap.items()}
        self.dmap_c = {d : self.get_chain(p) for d, (p,_,_) in self.dmap.items()}
        self.prio = None
    def __iter__(self):
        for p in self.dgm:
            yield p
    def set_prio(self, prio):
        if len(prio):
            self.prio = prio
        else:
            self.prio = None
    def remove_prio(self):
        self.prio = None
    def pair(self, p):
        return self.hom.pair(p.data)
    def is_paired(self, p):
        return self.pair(p) != self.hom.unpaired
    def get_birth(self, pt):
        return self.filt[pt.data]
    def get_death(self, pt):
        return self.filt[self.hom.pair(pt.data)]
    def get_chain(self, pt):
        return self.hom[self.hom.pair(pt.data)]
    def get_chain_idx(self, pt):
        chain = self.get_chain(pt)
        return [list(self.filt[c.index]) for c in chain]
    def get_pair(self, pt):
        return (self.get_birth(pt), self.get_death(pt))
    def match_death(self, t):
        return [(pbd,t.dmap[d]) for d,pbd in self.dmap.items() if d in t.dmap and t.dmap[d][1] != pbd[1]]
    def unmatched_death(self, t):
        return [pbd for d, pbd in self.dmap.items() if not d in t.dmap]
    def closest(self, q):
        dgm = self.dgm if self.prio is None else self.prio
        return min(dgm, key=lambda p: la.norm(q - np.array([p.birth,p.death])))

class GRFBarcodes:
    def __init__(self, field, cuts, max_stride=32, step=1, dim=1):
        self.N, _ = field.shape
        assert field.shape[1] == self.N
        self.field, self.cuts = field, cuts
        self.srange = list(range(max_stride, 0, -step)) # + [1]
        self.barcodes = [self.get_barcode(l) for l in tqdm(self.srange)]
        # self.barcodes = list(pmap(self.get_barcode, self.srange))
        self.full_barcode = self.barcodes[-1].dgms
        self.full_cuts = self.barcodes[-1].dgms_cut
        self.bnecks = self.get_bnecks(dim)
    def get_res(self, dim=1):
        return [[remove_inf(b[dim]) for b in bc.dgms_res] for bc in self.barcodes]
    def get_rel(self, dim=1):
        return [[remove_inf(b[dim]) for b in bc.dgms_rel] for bc in self.barcodes]
    def get_cuts(self, dim=1):
        return [remove_inf(b[dim]) for b in self.full_cuts]
    def down_sample(self, l):
        N, r = divmod(self.N, l)
        G = self.field[r//2:-r//2, r//2:-r//2] if r > 0 else self.field
        D = np.zeros((N, N), dtype=float)
        for j in range(N):
            for i in range(N):
                D[i, j] = G[i*l:(i+1)*l, j*l:(j+1)*l].sum() / (l ** 2)
        return D
    def get_barcode(self, l):
        G = self.down_sample(l if l > 0 else 1)
        return Barcode(G, self.cuts)
    def get_bnecks(self, dim=1):
        dgm_cut = self.get_cuts(dim)
        res_bc, rel_bc = self.get_res(dim), self.get_rel(dim)
        bneck_l = list(pmap(do_bottlenecks, res_bc+rel_bc, B=dgm_cut))
        return np.array(bneck_l).reshape(2, len(self.barcodes), len(self.cuts))
    def save(self, name, dir='data'):
        if not os.path.exists(dir):
            print('mkdir %s' % dir)
            os.mkdir(dir)
        fpath, i = os.path.join(dir, name), 0
        while os.path.exists('%s_%d.pkl' % (fpath, i)):
            i += 1
        fname = '%s_%d.pkl' % (fpath, i)
        print('saving %s' % fname)
        with open(fname, 'wb') as f:
            pkl.dump(self, f)

def load_barcodes(fname):
    print('loading %s' % fname)
    with open(fname, 'rb') as f:
        E = pkl.load(f)
    return E
