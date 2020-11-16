import numpy.linalg as la
import numpy as np

def gaussian(X, Y, c=[0., 0.], s=[0.5, 0.5]):
    return np.exp(-((X-c[0])**2 / (2*s[0]**2) + (Y-c[1])**2 / (2*s[1]**2)))

def mk_gauss(X, Y, args):
    return sum(w*gaussian(X, Y, c, r) for w, c, r in args)



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
