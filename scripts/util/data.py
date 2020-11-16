import numpy.linalg as la
import numpy as np

def gaussian(X, Y, c=[0., 0.], s=[0.5, 0.5]):
    return np.exp(-((X-c[0])**2 / (2*s[0]**2) + (Y-c[1])**2 / (2*s[1]**2)))

def mk_gauss(X, Y, args):
    return sum(w*gaussian(X, Y, c, r) for w, c, r in args)

COLOR = {'green' : (0, 205/255, 108/255),
        'blue' : (0, 154/255, 222/255),
        'purple' : (175/255, 88/255, 186/255),
        'orange' : (1, 198/255, 30/255)}

CUTS = [0.05, 0.3, 0.55, 0.8, 1.298]

N, WIDTH, HEIGHT = 64, 2, 1
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
