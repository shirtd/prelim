import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as la
import numpy as np
from mayavi.mlab import *
from mayavi.modules.surface import Surface

import dionysus as dio

# seed = np.random.randint(10000000)
seed = 4869361
np.random.seed(seed)
print(seed)

cuts = {'A' : (0.05, 0.3),
        'B' : (0.3, 0.55),
        'C' : (0.55, 0.8),
        'D' : (0.8, 1.3)}

COLOR = {'A' : (0, 205/255, 108/255),
        'B' : (0, 154/255, 222/255),
        'C' : (175/255, 88/255, 186/255),
        'D' : (1, 198/255, 30/255)}

PREV = {'A' : None, 'B' : 'A', 'C' : 'B', 'D' : 'C'}

def gaussian(X, Y, c=[0., 0.], s=[0.5, 0.5]):
    return np.exp(-((X-c[0])**2 / (2*s[0]**2) + (Y-c[1])**2 / (2*s[1]**2)))


N = 64
X, Y = np.meshgrid(np.linspace(-2,2,2*N), np.linspace(-1,1,1*N))

Gs = []

Gs.append(gaussian(X, Y, [-0.2, 0.2], [0.3, 0.3]))
Gs.append(0.5*gaussian(X, Y, [-1.3, -0.1], [0.15, 0.15]))
Gs.append(0.7*gaussian(X, Y, [-0.8, -0.4], [0.2, 0.2]))
Gs.append(0.8*gaussian(X, Y, [-0.8, -0], [0.4, 0.4]))

Gs.append(0.4*gaussian(X, Y, [0.6, 0.0], [0.4, 0.2]))

Gs.append(0.7*gaussian(X, Y, [1.25, 0.3], [0.25, 0.25]))

G = sum(Gs)

# def make_relative_sub_barcode():
mx = 1.297 # G.max()
lim = mx #max(max(p.death if p.death < np.inf else p.birth for p in d) for d in dgms if len(d))

def plot_barcode_super(axis, dgm):
    for i, (birth, death) in enumerate(dgm):
        i = 1 - i / len(dgm)
        death = -np.inf if death == np.inf else death
        for name, (a,b) in cuts.items():
            if a < death and birth <= b:
                axis.plot([birth, death], [i, i], c=COLOR[name], lw=5)
            elif death < a and birth > a and birth <= b:
                axis.plot([a, birth], [i, i], c=COLOR[name], lw=5)
            elif death > a and death < b and birth > b:
                axis.plot([death, b], [i, i], c=COLOR[name], lw=5)
            elif death <= a and b < birth:
                axis.plot([b, a], [i, i], c=COLOR[name], lw=5)
        if death == -np.inf:
            axis.plot([-0.05, 0.03], [i, i], c='black', linestyle='dotted')

def get_rel_dgm_super(name):
    filt = dio.fill_freudenthal(G, reverse=True)
    vmap = {v : filt[filt.index(dio.Simplex([v],0))].data for v in range(G.size)}
    rel = dio.Filtration([s for s in filt if all(vmap[v] < cuts[name][0] for v in s)])
    # rel = dio.Filtration([s for s in filt if all(vmap[v] > cuts[name][1] for v in s)])
    if name == 'A':
    # if name == 'D':
        hom = dio.homology_persistence(filt)
    else:
        hom = dio.homology_persistence(filt, relative=rel)
    dgms = dio.init_diagrams(hom, filt)
    return [np.array([[p.birth, p.death] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]

def get_res_dgm_super(name):
    filt = dio.fill_freudenthal(G, reverse=True)
    if name == 'A':
    # if name == 'D':
        res = filt
    else:
        vmap = {v : filt[filt.index(dio.Simplex([v],0))].data for v in range(G.size)}
        res = dio.Filtration([s for s in filt if all(vmap[v] >= cuts[name][0] for v in s)])
        # res = dio.Filtration([s for s in filt if all(vmap[v] <= cuts[name][1] for v in s)])
    hom = dio.homology_persistence(res)
    dgms = dio.init_diagrams(hom, res)
    return [np.array([[p.birth, p.death] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms]

def plot_barcodes(axs, dgms):
    for dim, dgm in enumerate(dgms):
        plot_barcode_super(axs[dim], dgm)

plt.ion()
fig, ax = plt.subplots(4, 3, figsize=(11, 6))

for a in ax:
    for aa in a:
        aa.set_xlim(0,1.39)
        aa.set_ylim(-0.2, 1.2)
        aa.get_yaxis().set_visible(False)
        aa.get_xaxis().set_visible(False)

# fig = plt.figure(1, figsize=(5,1.5))
# ax = plt.subplot(111)

ax[0,0].set_title(r"$\mathrm{H}_0$")
ax[0,1].set_title(r"$\mathrm{H}_1$")
ax[0,2].set_title(r"$\mathrm{H}_2$")
plt.tight_layout()

dgms = {name : get_res_dgm_super(name) for name in cuts.keys()}
# dgms = {name : get_rel_dgm_super(name) for name in cuts.keys()}

plot_barcodes(ax[0], dgms['A'])
plot_barcodes(ax[1], dgms['B'])
plot_barcodes(ax[2], dgms['C'])
plot_barcodes(ax[3], dgms['D'])

# for i, name in enumerate(['B','C','D']):
#     dgms = get_rel_dgm(name)
#     for dim, dgm in enumerate(dgms):
#         plot_barcode(ax[i+1,dim], dgm)
#


def make_sub_barcode():
    mx = 1.297 # G.max()

    plt.ion()
    fig = plt.figure(1, figsize=(5,1.5))
    ax = plt.subplot(111)

    filt = dio.fill_freudenthal(G)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)

    lim = 1.297 #max(max(p.death if p.death < np.inf else p.birth for p in d) for d in dgms if len(d))
    np_dgms = [np.array([[p.birth, p.death] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms ]

    # for name, (a,b) in cuts.items():
    #     ax.plot([b,b],[-1, 4], color=COLOR[name], linestyle='dotted')

    dgm = np_dgms[1]
    for i, (birth, death) in enumerate(dgm):
        for name, (a,b) in cuts.items():
            if a < birth and death <= b:
                ax.plot([birth, death], [i, i], c=COLOR[name], lw=5)
            elif birth < a and death > a and death <= b:
                ax.plot([a, death], [i, i], c=COLOR[name], lw=5)
            elif birth > a and birth < b and death > b:
                ax.plot([birth, b], [i, i], c=COLOR[name], lw=5)
            elif birth <= a and b < death:
                ax.plot([b, a], [i, i], c=COLOR[name], lw=5)
        if death == np.inf:
            ax.plot([lim, lim+0.1], [i, i], c='black', linestyle='dotted')

    ax.set_ylim(-1, 4)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

def make_super_barcode():
    mx = 1.297 # G.max()

    plt.ion()
    fig = plt.figure(1, figsize=(5,1.5))
    ax = plt.subplot(111)

    filt = dio.fill_freudenthal(G, reverse=True)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)

    lim = 1.297 #max(max(p.death if p.death < np.inf else p.birth for p in d) for d in dgms if len(d))
    np_dgms = [np.array([[p.birth, p.death if p.death < np.inf else -0.1] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms ]

    # for name, (a,b) in cuts.items():
    #     ax.plot([b,b],[-1, 4], color=COLOR[name], linestyle='dotted')

    for dim, dgm in enumerate(np_dgms):
        if len(dgm):
            for i, (birth, death) in enumerate(dgm):
                for name, (a,b) in cuts.items():
                    if a < death and birth <= b:
                        ax.plot([birth, death], [i, i], c=COLOR[name], lw=5)
                    elif death < a and birth > a and birth <= b:
                        ax.plot([a, birth], [i, i], c=COLOR[name], lw=5)
                    elif death > a and death < b and birth > b:
                        ax.plot([death, b], [i, i], c=COLOR[name], lw=5)
                    elif death <= a and b < birth:
                        ax.plot([b, a], [i, i], c=COLOR[name], lw=5)
                if death < 0:
                    ax.plot([-0.05, 0.03], [i, i], c='black', linestyle='dotted')

    ax.set_ylim(-1, 4)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

def make_diagram():
    mx = 1.297 # G.max()

    plt.ion()
    ax = plt.subplot(111)

    filt = dio.fill_freudenthal(G, reverse=True)
    hom = dio.homology_persistence(filt)
    dgms = dio.init_diagrams(hom, filt)

    lim = 1.297 #max(max(p.death if p.death < np.inf else p.birth for p in d) for d in dgms if len(d))
    np_dgms = [np.array([[p.birth, p.death if p.death < np.inf else -0.1] for p in d]) if len(d) else np.ndarray((0,2)) for d in dgms ]

    ax.plot([0,lim], [0,lim], c='black', alpha=0.5, zorder=1)
    ax.plot([0, lim], [0, 0], c='black', alpha=0.5, linestyle='dashed', zorder=0)
    # ax.plot([lim, lim], [lim, 0], c='black', alpha=0.5, linestyle='dashed', zorder=0)

    # ax.plot([0,1.25*lim], [0,1.25*lim], c='black', alpha=0.5, zorder=1)
    # ax.plot([0, lim], [lim, lim], c='black', alpha=0.5, linestyle='dashed', zorder=0)
    # ax.plot([lim, lim], [lim, 1.25*lim], c='black', alpha=0.5, linestyle='dashed', zorder=0)


    for name, (a,b) in cuts.items():
        ax.plot([b, lim], [b, b], c=COLOR[name], alpha=0.5, linestyle='dashed', zorder=0)
            # ax.plot([la, la], [la, 1.25*lim], c=c, alpha=0.5, linestyle='dotted', zorder=0)
        # ax.plot([lb, lb], [la, la], c=c, alpha=0.5, linestyle='dotted', zorder=0)

    for dim, dgm in enumerate(np_dgms):
        if len(dgm):
            for birth, death in dgm:
                c = 'black'
                for name, (a,b) in cuts.items():
                    if a <= birth and birth < b:
                        c = COLOR[name]
                        break
                ax.scatter(birth, death, s=10, color=c, zorder=2)#, label='dimension %d' % dim)
            # ax.scatter(dgm[:,0], dgm[:,1], s=10, zorder=2, label='dimension %d' % dim)

    # ax.legend(loc='lower right')


def make_surf():
    s0 = surf(X.T, Y.T, G)
    s0.visible = False
    ctl = s0.parent
    scene = ctl.parent.parent.parent.parent.scene

    S = {}

    for name, (a, b) in cuts.items():
        S[name] = Surface()
        S[name].name = name
        S[name].enable_contours = True
        S[name].contour.filled_contours = True
        ctl.add_child(S[name])
        S[name].contour.minimum_contour = a
        S[name].contour.maximum_contour = b
        S[name].actor.property.lighting = False
        S[name].actor.mapper.scalar_visibility = False
        S[name].actor.property.color = COLOR[name]
        S[name].actor.property.opacity = 0.5

    S['A'].actor.property.backface_culling = True


    scene.parallel_projection = True
    scene.background = (1,1,1)

    view(-11.800830502323867, 80.88795919149756, 9.035877007511106,
        np.array([-1.00787402,  1.01587307,  0.6490444 ]))
    scene.camera.parallel_scale = 1.5973342386092595

    # plt.ion()
    # # ax = plt.subplot(111)
    # fig2 = plt.figure(2)
    # ax2 = plt.subplot(projection='3d')
    # ax2.set_xlim(-2,2); ax2.set_ylim(-2,2)
    # ax2.plot_surface(X, Y, G, rcount=64, ccount=2*64, cmap='viridis')

# if __name__ == "__main__":
#     # make_super_barcode()
#     # make_surf()
