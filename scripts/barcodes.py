import matplotlib.pyplot as plt
from util.persist import *
import os, sys

plt.ion()
# fig, ax = plt.subplots(4, 3, figsize=(11, 6))
fig, ax = plt.subplots(3, 3, figsize=(10, 4))

for a in ax:
    for aa in a:
        aa.set_xlim(0,1.39)
        aa.set_ylim(-0.2, 1.2)
        aa.get_yaxis().set_visible(False)
        aa.get_xaxis().set_visible(False)

# fig = plt.figure(1, figsize=(5,1.5))
# ax = plt.subplot(111)

# ax[0,0].set_title(r"$\mathrm{H}_0$")
# ax[0,1].set_title(r"$\mathrm{H}_1$")
# ax[0,2].set_title(r"$\mathrm{H}_2$")
# plt.tight_layout()

def plot_barcodes(axs, dgms, filt_t='sub', lw=5):
    for dim, dgm in enumerate(dgms):
        if filt_t == 'sub':
            plot_barcode_sub(axs[dim], dgm, lw)
        elif filt_t == 'super':
            plot_barcode_super(axs[dim], dgm, lw)

def do_plot(axs, typ, flt, G, cut_map):
    if typ == 'sub':
        if flt == 'res':
            dgms = {name : get_res_dgm_sub(G, name) for name in cut_map.keys()}
        elif flt == 'rel':
            dgms = {name : get_rel_dgm_sub(G, name) for name in cut_map.keys()}
    elif typ == 'super':
        if flt == 'res':
            dgms = {name : get_res_dgm_super(G, name) for name in cut_map.keys()}
        elif flt == 'rel':
            dgms = {name : get_rel_dgm_super(G, name) for name in cut_map.keys()}

if __name__ == "__main__":
    G = mk_gauss(X, Y, GAUSS_ARGS)

    THRESH = 4 * np.sqrt(2 * (2 / len(G)) ** 2)

    rel = {name : get_rel_dgm_sub(G, name) for name in CUT_MAP.keys()}
    res = {name : get_res_dgm_sub(G, name) for name in CUT_MAP.keys()}

    ax[0,0].set_title(r"$\mathrm{H}_0$ full")
    ax[0,1].set_title(r"$\mathrm{H}_0$ restricted")
    ax[0,2].set_title(r"$\mathrm{H}_0$ relative")

    ax[1,0].set_title(r"$\mathrm{H}_1$ full")
    ax[1,1].set_title(r"$\mathrm{H}_1$ restricted")
    ax[1,2].set_title(r"$\mathrm{H}_1$ relative")

    ax[2,0].set_title(r"$\mathrm{H}_2$ full")
    ax[2,1].set_title(r"$\mathrm{H}_2$ restricted")
    ax[2,2].set_title(r"$\mathrm{H}_2$ relative")

    plt.tight_layout()

    plot_barcode_sub(ax[0,0], res['A'][0], 5)
    plot_barcode_sub(ax[1,0], res['A'][1], 5)
    plot_barcode_sub(ax[2,0], res['A'][2], 5)

    plot_barcode_sub(ax[0,1], [[b,d] for b,d in res['C'][0] if d - b > THRESH], 5, 4)
    plot_barcode_sub(ax[1,1], res['C'][1], 5)
    plot_barcode_sub(ax[2,1], res['C'][2], 5)

    plot_barcode_sub(ax[0,2], rel['C'][0], 5)
    plot_barcode_sub(ax[1,2], rel['C'][1], 5, 4, 2)
    plot_barcode_sub(ax[2,2], rel['C'][2], 5, 4)

    # TYPE = 'sub' # 'super'
    # FILT = 'res' # 'rel' #
    #
    # if TYPE == 'sub':
    #     if FILT == 'res':
    #         dgms = {name : get_res_dgm_sub(G, name) for name in CUT_MAP.keys()}
    #     elif FILT == 'rel':
    #         dgms = {name : get_rel_dgm_sub(G, name) for name in CUT_MAP.keys()}
    # elif TYPE == 'super':
    #     if FILT == 'res':
    #         dgms = {name : get_res_dgm_super(G, name) for name in CUT_MAP.keys()}
    #     elif FILT == 'rel':
    #         dgms = {name : get_rel_dgm_super(G, name) for name in CUT_MAP.keys()}
    #
    # plot_barcodes(ax[0], dgms['A'], TYPE)
    # plot_barcodes(ax[1], dgms['B'], TYPE)
    # plot_barcodes(ax[2], dgms['C'], TYPE)
    # plot_barcodes(ax[3], dgms['D'], TYPE)

    DIR = os.path.join('figures', 'barcodes')
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    # fname = os.path.join(DIR, '%s_%s.png' % (TYPE, FILT))
    fname = os.path.join(DIR, 'res_rel.png')
    print('saving %s' % fname)
    plt.savefig(fname, dpi=300)

    # for i, name in enumerate(['B','C','D']):
    #     dgms = get_rel_dgm(name)
    #     for dim, dgm in enumerate(dgms):
    #         plot_barcode(ax[i+1,dim], dgm)
