import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as la
import numpy as np
import os, sys

from util.plot import SurfacePlot
from util.data import mk_gauss

import dionysus as dio

SEED = 4869361 # np.random.randint(10000000) #
print('seed: %d' % SEED)
np.random.seed(SEED)

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

SURF_ARGS = {   'A' : {'min' : CUTS[0], 'max' : CUTS[1],    'color' : COLOR['green'],   'opacity' : 0.5,    'backface_culling' : True},
                'B' : {'min' : CUTS[1], 'max' : CUTS[2],    'color' : COLOR['blue'],    'opacity' : 0.5},
                'C' : {'min' : CUTS[2], 'max' : CUTS[3],    'color' : COLOR['purple'],  'opacity' : 0.5},
                'D' : {'min' : CUTS[3], 'max' : CUTS[4],    'color' : COLOR['orange'],  'opacity' : 0.5}}

CONT_ARGS = {   'A_c' : {'scalar' : [CUTS[1]], 'color' : COLOR['green']},
                'B_c' : {'scalar' : [CUTS[2]], 'color' : COLOR['blue']},
                'C_c' : {'scalar' : [CUTS[3]], 'color' : COLOR['purple']}}

VIEW = {'default' : 'side',
        'side' : {  'view' : (-11.800830502323867, 80.88795919149756, 9.035877007511106,
                        np.array([-1.00787402,  1.01587307,  0.6490444])),
                    'zoom' : 1.6, 'roll' : -89},
        'top' : {   'view' : (0.0, 0.0, 8.291977298839994,
                        np.array([-1.00648859,  1.05123171,  0.67399999])),
                    'zoom' : 1.6, 'roll' : -80}}

def ass1(surf, dir=os.path.join('figures', 'surf')):
    if not os.path.exists(dir):
        os.mkdir(dir)
    surf.focus_high('C')
    surf.reset_view('side')
    surf.save(os.path.join(dir, 'ass1_C_side.png'))
    surf.reset_view('top')
    surf.save(os.path.join(dir, 'ass1_C_top.png'))
    surf.focus_low('D')
    surf.reset_view('side')
    surf.save(os.path.join(dir, 'ass1_D_side.png'))
    surf.reset_view('top')
    surf.save(os.path.join(dir, 'ass1_D_top.png'))

def ass2(surf, dir=os.path.join('figures', 'surf')):
    if not os.path.exists(dir):
        os.mkdir(dir)
    surf.focus_high('B')
    surf.reset_view('side')
    surf.save(os.path.join(dir, 'ass2_B_side.png'))
    surf.reset_view('top')
    surf.save(os.path.join(dir, 'ass2_B_top.png'))
    surf.focus_low('C')
    surf.reset_view('side')
    surf.save(os.path.join(dir, 'ass2_C_side.png'))
    surf.reset_view('top')
    surf.save(os.path.join(dir, 'ass2_C_top.png'))

if __name__ == "__main__":
    DIR = os.path.join('figures', 'surf')
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    G = mk_gauss(X, Y, GAUSS_ARGS)
    surf = SurfacePlot(X, Y, G, SURF_ARGS, CONT_ARGS, VIEW)

    surf.reset_view('side')
    surf.save(os.path.join(DIR, 'side.png'))
    surf.reset_view('top')
    surf.save(os.path.join(DIR, 'top.png'))

    ass1(surf, DIR)
    ass2(surf, DIR)
