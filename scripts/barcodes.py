import matplotlib.pyplot as plt
from util.persist import *
import os, sys

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

    TYPE = 'sub' # 'super'
    FILT = 'res' # 'rel' #

    if TYPE == 'sub':
        if FILT == 'res':
            dgms = {name : get_res_dgm_sub(G, name) for name in CUT_MAP.keys()}
        elif FILT == 'rel':
            dgms = {name : get_rel_dgm_sub(G, name) for name in CUT_MAP.keys()}
    elif TYPE == 'super':
        if FILT == 'res':
            dgms = {name : get_res_dgm_super(G, name) for name in CUT_MAP.keys()}
        elif FILT == 'rel':
            dgms = {name : get_rel_dgm_super(G, name) for name in CUT_MAP.keys()}

    plot_barcodes(ax[0], dgms['A'], TYPE)
    plot_barcodes(ax[1], dgms['B'], TYPE)
    plot_barcodes(ax[2], dgms['C'], TYPE)
    plot_barcodes(ax[3], dgms['D'], TYPE)

    DIR = os.path.join('figures', 'barcodes')
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    fname = os.path.join(DIR, '%s_%s.png' % (TYPE, FILT))
    print('saving %s' % fname)
    plt.savefig(fname)

    # for i, name in enumerate(['B','C','D']):
    #     dgms = get_rel_dgm(name)
    #     for dim, dgm in enumerate(dgms):
    #         plot_barcode(ax[i+1,dim], dgm)
