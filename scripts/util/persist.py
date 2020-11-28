from util.data import *
import dionysus as dio

# def make_relative_sub_barcode():
mx = 1.3 # G.max()
lim = mx #max(max(p.death if p.death < np.inf else p.birth for p in d) for d in dgms if len(d))

CUT_MAP = {'A' : (CUTS[0], CUTS[1]),
        'B' : (CUTS[1], CUTS[2]),
        'C' : (CUTS[2], CUTS[3]),
        'D' : (CUTS[3], mx)}

CMAP = {'A' : COLOR['green'],
        'B' : COLOR['blue'],
        'C' : COLOR['purple'],
        'D' : COLOR['yellow']}

def plot_barcode_sub(axis, dgm, lw=5, N=None, offset=0):
    N = len(dgm) if N is None else N
    for i, (birth, death) in enumerate(dgm):
        i = 1 - (i + offset) / (N - 1)
        for name, (a,b) in CUT_MAP.items():
            if a < birth and death <= b:
                axis.plot([birth, death], [i, i], c=CMAP[name], lw=lw)
            elif birth < a and death > a and death <= b:
                axis.plot([a, death], [i, i], c=CMAP[name], lw=lw)
            elif birth > a and birth < b and death > b:
                axis.plot([birth, b], [i, i], c=CMAP[name], lw=lw)
            elif birth <= a and b < death:
                axis.plot([b, a], [i, i], c=CMAP[name], lw=lw)
        if death == np.inf:
            axis.plot([1.32, 1.39], [i, i], c='black', linestyle='dotted',  lw=lw/4)
        # ax.set_ylim(-1, 4)
        axis.get_yaxis().set_visible(False)

def plot_barcode_super(axis, dgm, lw=5):
    for i, (birth, death) in enumerate(dgm):
        i = 1 - i / len(dgm)
        death = -np.inf if death == np.inf else death
        for name, (a,b) in CUT_MAP.items():
            if a < death and birth <= b:
                axis.plot([birth, death], [i, i], c=CMAP[name], lw=lw)
            elif death < a and birth > a and birth <= b:
                axis.plot([a, birth], [i, i], c=CMAP[name], lw=lw)
            elif death > a and death < b and birth > b:
                axis.plot([death, b], [i, i], c=CMAP[name], lw=lw)
            elif death <= a and b < birth:
                axis.plot([b, a], [i, i], c=CMAP[name], lw=lw)
        if death == -np.inf:
            axis.plot([-0.05, 0.03], [i, i], c='black', linestyle='dotted', lw=lw/2)

def get_rel_dgm_sub(G, name):
    filt = dio.fill_freudenthal(G)
    vmap = {v : filt[filt.index(dio.Simplex([v],0))].data for v in range(G.size)}
    rel = dio.Filtration([s for s in filt if all(vmap[v] < CUT_MAP[name][0] for v in s)])
    if name == 'A':
        hom = dio.homology_persistence(filt)
    else:
        hom = dio.homology_persistence(filt, relative=rel)
    dgms = dio.init_diagrams(hom, filt)
    return [np.array([[p.birth, p.death] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]

def get_rel_dgm_super(G, name):
    filt = dio.fill_freudenthal(G, reverse=True)
    vmap = {v : filt[filt.index(dio.Simplex([v],0))].data for v in range(G.size)}
    rel = dio.Filtration([s for s in filt if all(vmap[v] < CUT_MAP[name][0] for v in s)])
    # rel = dio.Filtration([s for s in filt if all(vmap[v] > CUT_MAP[name][1] for v in s)])
    if name == 'A':
    # if name == 'D':
        hom = dio.homology_persistence(filt)
    else:
        hom = dio.homology_persistence(filt, relative=rel)
    dgms = dio.init_diagrams(hom, filt)
    return [np.array([[p.birth, p.death] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]

def get_res_dgm_sub(G, name):
    filt = dio.fill_freudenthal(G)
    if name == 'A':
        res = filt
    else:
        vmap = {v : filt[filt.index(dio.Simplex([v],0))].data for v in range(G.size)}
        res = dio.Filtration([s for s in filt if all(vmap[v] >= CUT_MAP[name][0] for v in s)])
    hom = dio.homology_persistence(res)
    dgms = dio.init_diagrams(hom, res)
    return [np.array([[p.birth, p.death] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]

def get_res_dgm_super(G, name):
    filt = dio.fill_freudenthal(G, reverse=True)
    if name == 'A':
    # if name == 'D':
        res = filt
    else:
        vmap = {v : filt[filt.index(dio.Simplex([v],0))].data for v in range(G.size)}
        res = dio.Filtration([s for s in filt if all(vmap[v] >= CUT_MAP[name][0] for v in s)])
        # res = dio.Filtration([s for s in filt if all(vmap[v] <= CUT_MAP[name][1] for v in s)])
    hom = dio.homology_persistence(res)
    dgms = dio.init_diagrams(hom, res)
    return [np.array([[p.birth, p.death] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]
