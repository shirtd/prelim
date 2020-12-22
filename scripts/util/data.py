import numpy.linalg as la
import numpy as np

from numpy.random import normal, randint, rand
from numpy.fft import fft2, ifft2
import numpy as np
import numpy
import scipy.fftpack

def gaussian(X, Y, c=[0., 0.], s=[0.5, 0.5]):
    return np.exp(-((X-c[0])**2 / (2*s[0]**2) + (Y-c[1])**2 / (2*s[1]**2)))

def mk_gauss(X, Y, args):
    return sum(w*gaussian(X, Y, c, r) for w, c, r in args)

def grf(alpha=-3.0, m=128, flag_normalize=True):
    size = int(np.sqrt(m))
    k_ind = np.mgrid[:size, :size] - int((size + 1) / 2)
    k_idx = scipy.fftpack.fftshift(k_ind)
    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = numpy.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, alpha / 4.0)
    amplitude[0,0] = 0
    # Draws a complex gaussian random noise with normal (circular) distribution
    noise = np.random.normal(size = (size, size)) + 1j * np.random.normal(size = (size, size))
    gfield = np.fft.ifft2(noise * amplitude).real # To real space
    return (gfield - gfield.min()) / (gfield.max() - gfield.min())

COLOR = {'green'    :   (0, 205/255, 108/255),
        'blue'      :   (0, 154/255, 222/255),
        'purple'    :   (175/255, 88/255, 186/255),
        'yellow'    :   (1, 198/255, 30/255),
        'orange'    :   (242/255, 133/255, 34/255),
        'gray'      :   (160/225, 177/225, 186/225),
        'red'       :   (255/255, 31/255, 91/255),
        'brown'     :   (166/255, 118/255, 29/255),
        'black'     :   (0, 0, 0)}

CUTS = [0.05, 0.3, 0.55, 0.8, 1.298]

N, WIDTH, HEIGHT = 64, 2, 1
# N, WIDTH, HEIGHT = 512, 2, 1
X_RNG = np.linspace(-WIDTH,WIDTH,WIDTH*N)
Y_RNG = np.linspace(-HEIGHT,HEIGHT,HEIGHT*N)
X, Y = np.meshgrid(X_RNG, Y_RNG)

GAUSS_ARGS = [(1, [-0.2, 0.2], [0.3, 0.3]),
            (0.5, [-1.3, -0.1], [0.15, 0.15]),
            (0.7, [-0.8, -0.4], [0.2, 0.2]),
            (0.8, [-0.8, -0], [0.4, 0.4]),
            (0.4, [0.6, 0.0], [0.4, 0.2]),
            (0.7, [1.25, 0.3], [0.25, 0.25])]


# Gs = []
# Gs.append(gaussian(X, Y, [-0.2, 0.2], [0.3, 0.3]))
# Gs.append(0.5*gaussian(X, Y, [-1.3, -0.1], [0.15, 0.15]))
# Gs.append(0.7*gaussian(X, Y, [-0.8, -0.4], [0.2, 0.2]))
# Gs.append(0.8*gaussian(X, Y, [-0.8, -0], [0.4, 0.4]))
#
# Gs.append(0.4*gaussian(X, Y, [0.6, 0.0], [0.4, 0.2]))
#
# Gs.append(0.7*gaussian(X, Y, [1.25, 0.3], [0.25, 0.25]))

# G = sum(Gs)
