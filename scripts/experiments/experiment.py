import matplotlib.pyplot as plt
from util.bottleneck import *
from util.data import grf, gaussian, COLOR
import argparse

if __name__ == "__main__":
    # SEED = 817 # np.random.randint(1024) # 3042755 # 344 # 95 #  928 #
    SEED = np.random.randint(1024) # 3042755 # 344 # 95 #  928 #
    print('seed: %d' % SEED)
    np.random.seed(SEED)

    def ripple(x, y, f=1, l=1, w=1):
        t = la.norm(np.stack((x, y), axis=2), axis=2) + 1/12
        t[t > 1] = 1.
        return (1 - t) - np.exp(-t / l) * np.cos(2*np.pi*f*(1-t) * np.sin(2*np.pi*w*t))

    EXP = -3
    # EXP = -3.5
    # N, WIDTH, HEIGHT = 256, 0.5, 0.5
    _f, _l, _w = 1, 1, 1
    # _f, _l, _w = 5, 0.5, 0.5
    N, WIDTH, HEIGHT = 1024, 1, 1
    X_RNG = np.linspace(-WIDTH,WIDTH,int(WIDTH*N))
    Y_RNG = np.linspace(-HEIGHT,HEIGHT,int(HEIGHT*N))
    X, Y = np.meshgrid(X_RNG, Y_RNG)

    DIM = 1
    LOAD = False
    SAVE = True

    if len(sys.argv) > 1:
        LOAD = True
        fname = sys.argv[1]
        self = load_barcodes(fname)
    else:
        LOAD = False

        # field = grf(EXP, N*N)
        # field = (0.4 * gaussian(X, Y, [0,0], [0.4, 0.4])
        #             + 0.6 * gaussian(X, Y, [0,0], [0.1, 0.1])
        #             + ripple(X, Y, _f, _l, _w)
        #                 * (1 + grf(EXP, N*N)))
        # field = (0.7 * gaussian(X, Y, [0,0], [0.7, 0.7])
                    # + 0.6 * gaussian(X, Y, [0,0], [0.2, 0.2])
        # field = ripple(X, Y, _f, _l, _w) * grf(EXP, N*N)
        field = ripple(X, Y, _f, _l, _w) * (1 + grf(EXP, N*N))

        field = (field - field.min()) / (field.max() - field.min())
        # cuts = [0.3, 0.45, 0.6, 0.75]
        # cuts = [0.2, 0.4, 0.6, 0.8]
        cuts = [0.3, 0.5, 0.7]
        # cuts = [0.25, 0.5, 0.75]

        plt.ion()
        # # plt.imshow(field)
        # ax = plt.subplot(projection='3d')
        # ax.plot_surface(X, Y, field)

        self = GRFBarcodes(field, cuts, max_stride=16)

        # name = '%d_%d%d' % (SEED, N, EXP)
        # name = '%d_%d-%d' % (N, _f, _l, _w)
        name = '%d_%d%d_%d-%d' % (SEED, N, EXP, _f, _w)
        if SAVE:
            self.save(name)

    plt.ion()
    fig, ax = plt.subplots(1,len(self.cuts),
                            sharex=True, sharey=True,
                            figsize=(3*len(self.cuts)+1,4))

    x_rng = [N // l for l in self.srange]
    d_rng = np.array([np.sqrt(2 * (2/(N // l)) ** 2) for l in self.srange]) / 2
    delta = np.sqrt(2 * (2/N) ** 2) / 2

    for j,c in enumerate(self.cuts):
        ax[j].cla()
        ax[j].plot(d_rng, 2*d_rng, alpha=0.5, label="Expected", color='black')
        ax[j].plot(d_rng, self.bnecks[0,:,j], alpha=1, label="Restricted", color=COLOR['red'])
        ax[j].plot(d_rng, self.bnecks[1,:,j], alpha=1, label="Relative", color=COLOR['blue'])
        ax[j].invert_xaxis()

    plt.legend()
    ax[0].set_title(r"$\omega = %0.1f$" % self.cuts[0])
    ax[1].set_title(r"$\omega = %0.1f$" % self.cuts[1])
    ax[2].set_title(r"$\omega = %0.1f$" % self.cuts[2])
    # ax[3].set_title(r"$\omega = %0.1f$" % self.cuts[3])
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.suptitle(r"$\mathrm{H}_1$ Bottleneck Distance to Truncated Diagram (%dx%d)" % (N, N))
    plt.suptitle(r"$\mathrm{H}_1$ Bottleneck Distance to Truncated Diagram ($\delta \approx %0.1e$)" % delta)
    # plt.suptitle(r"$\mathrm{H}_1$ Bottleneck Distance to Full Diagram (%dx%d)" % (N, N))
    plt.tight_layout()

    # plt.xlabel("Grid Size", labelpad=10)
    plt.xlabel("$\delta$", labelpad=10)
    plt.ylabel("Bottleneck Distance", labelpad=20)

    if not LOAD:
        fig_dir = os.path.join('figures', 'bottleneck')
        if not os.path.exists(fig_dir):
            print('mkdir %s' % fig_dir)
            os.mkdir(fig_dir)
        fpath, i = os.path.join(fig_dir, name), 0
        while os.path.exists('%s_%d.png' % (fpath, i)):
            i += 1
        fname = '%s_%d.png' % (fpath, i)
        print('saving %s' % fname)
        plt.savefig(fname, dpi=200)

    plt.pause(10)
