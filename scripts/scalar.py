import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.linalg as la
import numpy as np
from mayavi.mlab import *
from mayavi.modules.surface import Surface

# seed = np.random.randint(10000000)
seed = 4869361
np.random.seed(seed)
print(seed)

def gaussian(X, Y, c=[0., 0.], s=[0.5, 0.5]):
    return np.exp(-((X-c[0])**2 / (2*s[0]**2) + (Y-c[1])**2 / (2*s[1]**2)))


N = 64
X, Y = np.meshgrid(np.linspace(-2,2,2*N), np.linspace(-1,1,1*N))

Gs = []

Gs.append(gaussian(X, Y, [-0.2, 0.2], [0.3, 0.3]))
Gs.append(0.5*gaussian(X, Y, [-1.3, -0.1], [0.15, 0.15]))
Gs.append(0.7*gaussian(X, Y, [-0.8, -0.4], [0.2, 0.2]))
Gs.append(0.8*gaussian(X, Y, [-0.8, -0], [0.4, 0.4]))

# Gs.append(0.3*gaussian(X, Y, [1.15, -0.3], [0.15, 0.15]))
# Gs.append(0.4*gaussian(X, Y, [0.25, -0.4], [0.2, 0.2]))
# Gs.append(0.5*gaussian(X, Y, [0.6, 0.15], [0.25, 0.25]))

Gs.append(0.4*gaussian(X, Y, [0.6, 0.0], [0.4, 0.2]))


Gs.append(0.7*gaussian(X, Y, [1.25, 0.3], [0.25, 0.25]))
# Gs.append(0.8*gaussian(X, Y, [0.3, 0.3], [0.05, 0.07]))
# Gs.append(0.72*gaussian(X, Y, [0.1, -0.3], [0.07, 0.07]))
# Gs.append(0.8*gaussian(X, Y, [0.2, -0.1], [0.08, 0.07]))
# Gs.append(0.6*gaussian(X, Y, [-0.3, -0.15], [0.05, 0.05]))
# Gs.append(0.75*gaussian(X, Y, [-0.2, 0.15], [0.08, 0.14]))
# Gs.append(1.2*gaussian(X, Y, [0.1, 0.15], [0.1, 0.1]))
#
#
# Gs.append(0.6*gaussian(X, Y, [-0.3, -0.7], [0.1, 0.1]))
# Gs.append(0.65*gaussian(X, Y, [0, -0.7], [0.3, 0.1]))

G = sum(Gs)

F = np.array

s0 = surf(X.T, Y.T, G)
s0.visible = False
ctl = s0.parent
scene = ctl.parent.parent.parent.parent.scene

S = {}
cuts = [('A', 0.05, 0.3),
        ('B', 0.3, 0.55),
        ('C', 0.55, 0.8),
        ('D', 0.8, 1.298)]

for name, a, b in cuts:
    S[name] = Surface()
    S[name].name = name
    S[name].enable_contours = True
    S[name].contour.filled_contours = True
    ctl.add_child(S[name])
    S[name].contour.minimum_contour = a
    S[name].contour.maximum_contour = b
    S[name].actor.property.lighting = False
    S[name].actor.mapper.scalar_visibility = False

S['A'].actor.property.color = (0, 205/255, 108/255)
S['B'].actor.property.color = (0, 154/255, 222/255)
S['C'].actor.property.color = (175/255, 88/255, 186/255)
S['D'].actor.property.color = (1, 198/255, 30/255)

S['A'].actor.property.backface_culling = True
S['A'].actor.property.opacity = 0.5
S['B'].actor.property.opacity = 0.5
S['C'].actor.property.opacity = 0.5
S['D'].actor.property.opacity = 0.5


scene.parallel_projection = True
scene.background = (1,1,1)

view(-11.800830502323867,
    80.88795919149756,
    9.035877007511106,
    np.array([-1.00787402,  1.01587307,  0.6490444 ]))
scene.camera.parallel_scale = 1.5973342386092595

# scene.camera.roll(-89.16252353115732)

# scene.camera.view_angle = 30
# scene.camera.view_shear = np.array([0., 0., 1.])
#
# scene.camera.distance =  9.035877007511177
# scene.camera.view_plane_normal = (0.9139941936760938, -0.37565773949872383, 0.1532836477941985)
# scene.camera.view_up = np.array([-0.15430236,  0.02757146,  0.9876389 ])

# .actor.mapper.scalar_visibility = False


# c.trait_set(number_of_contours=20)

# plt.ion()
# # ax = plt.subplot(111)
# fig2 = plt.figure(2)
# ax2 = plt.subplot(projection='3d')
# ax2.set_xlim(-2,2); ax2.set_ylim(-2,2)
# ax2.plot_surface(X, Y, G, rcount=64, ccount=2*64, cmap='viridis')


# ax2.contourf(X, Y, G, [0.3], colors=['red'])
