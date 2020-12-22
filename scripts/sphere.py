import numpy as np
from mayavi import mlab
from util.data import *
import os

DIR = os.path.join('figures', 'spheres2')
if not os.path.exists(DIR):
    os.mkdir(DIR)


# VIEWID = 'front'
# VIEW = (0.0, 0.0, 12.764387888094552, np.array([0., 0., 0.]))

VIEWID = 'side'
VIEW = (-43., 35., 13., np.array([0., 0., 0.]))

SAVE = False
BALLS = False
r = 1.2 #1 # 2*np.sqrt(2) / 3
phi, theta = np.mgrid[0:np.pi:101j, 0:2 * np.pi:101j]

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)


P = np.array([[1, 0, -1/np.sqrt(2)],
                [0, -1, 1/np.sqrt(2)],
                [0, 1, 1/np.sqrt(2)],
                [-1, 0, -1/np.sqrt(2)]])

points = mlab.points3d(P[:,0], P[:,1], P[:,2], color=(0,0,0), scale_factor=0.2)
points.actor.property.lighting = False
points.glyph.glyph_source.glyph_source.phi_resolution = 32
points.glyph.glyph_source.glyph_source.theta_resolution = 32


# P = np.array([[-1.,0.,0.],[1.,0.,0.]])
# E = [0,1]
#
# line = mlab.plot3d(P[E,0], P[E,1], P[E,2], color=COLOR['blue'])
# line.actor.property.lighting = False
#
# points = mlab.points3d(P[:,0], P[:,1], P[:,2], color=(0,0,0), scale_factor=0.2)
# points.actor.property.lighting = False
# points.glyph.glyph_source.glyph_source.phi_resolution = 32
# points.glyph.glyph_source.glyph_source.theta_resolution = 32

gcf = mlab.gcf()
scene = gcf.scene
scene.background = (1,1,1)
mlab.view(*VIEW)

if SAVE:
    mlab.savefig(os.path.join(DIR, 'dump-%s.png' % VIEWID), size=(10,10))
    mlab.savefig(os.path.join(DIR, 'dump-%s.png' % VIEWID), size=(10,10))

if BALLS:
    surfs = [mlab.mesh(x+u, y+v, z+w, color=COLOR['red'], opacity=0.3) for u,v,w in P]
    for surf in surfs:
        surf.actor.property.lighting = False

    if SAVE:
        mlab.savefig(os.path.join(DIR, 'balls-%s.png' % VIEWID), size=(2000,2000))

lines = [mlab.plot3d(P[e,0], P[e,1], P[e,2], color=COLOR['blue']) for e in ([0,1,2],[1,3,0],[0,2,3])]
for line in lines:
    line.actor.property.lighting = False

if SAVE:
    mlab.savefig(os.path.join(DIR, 'edges-%s.png' % VIEWID), size=(2000,2000))

T = [[0,1,2],[1,2,3],[0,1,3],[0,2,3]]
tris = mlab.triangular_mesh(P[:,0], P[:,1], P[:,2], T, color=COLOR['blue'], opacity=0.5)
tris.actor.property.lighting = False

if SAVE:
    mlab.savefig(os.path.join(DIR, 'tris-%s.png' % VIEWID), size=(2000,2000))

tris.actor.property.opacity = 1

if SAVE:
    mlab.savefig(os.path.join(DIR, 'poly-%s.png' % VIEWID), size=(2000,2000))

P = np.array([[np.cos(2 * np.pi* i / 3), np.sin(2 * np.pi * i / 3), 0] for i in range(3)])
E = [[0,1], [1,2], [2,0]]

lines = [mlab.plot3d(P[e,0], P[e,1], P[e,2], color=COLOR['blue']) for e in E]
for line in lines:
    line.actor.property.lighting = False

points = mlab.points3d(P[:,0], P[:,1], P[:,2], color=(0,0,0), scale_factor=0.2)
points.actor.property.lighting = False
points.glyph.glyph_source.glyph_source.phi_resolution = 32
points.glyph.glyph_source.glyph_source.theta_resolution = 32

tris = mlab.triangular_mesh(P[:,0], P[:,1], P[:,2], [[0,1,2]], color=COLOR['blue'], opacity=0.5)
tris.actor.property.lighting = False


# T = [[0,1,2],[1,2,3],[0,1,3],[0,2,3]]
# tris = mlab.triangular_mesh(P[:,0], P[:,1], P[:,2], T, color=COLOR['blue'])
# tris.actor.property.lighting = False
