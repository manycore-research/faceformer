from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Extend.TopologyUtils import TopologyExplorer, discretize_edge
import numpy as np

def randnum(low, high):
    return np.random.rand() * (high - low) + low

# generate a random camera
def generate_random_camera_pos(seed):
    np.random.seed(seed)
    focus = randnum(3, 5)
    radius = randnum(1.25, 1.5) # distance of camera to origin
    phi = randnum(22.5, 67.5) # longitude, elevation of camera
    theta = randnum(0, 360) # latitude, rotation around z-axis
    return focus, pose_spherical(theta, phi, radius)

def pose_spherical(theta, phi, radius):
    def trans_t(t): return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    def rot_phi(phi): return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    def rot_theta(th): return np.array([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    c2w = trans_t(radius)
    c2w = rot_phi(np.deg2rad(phi)) @ c2w
    c2w = rot_theta(np.deg2rad(theta)) @ c2w
    c2w = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ c2w
    return c2w



def project_shapes(shapes, args):
    location = args.location
    direction = args.direction
    focus = args.focus

    hlr = HLRBRep_Algo()

    if isinstance(shapes, list):
        for shape in shapes:
            hlr.Add(shape)
    else:
        hlr.Add(shapes)
    ax = gp_Ax2(gp_Pnt(*location), gp_Dir(*direction))

    if args.pose is not None:
        pose = args.pose
        ax = gp_Ax2(gp_Pnt(*pose[:3, -1]), gp_Dir(*pose[:3, -2]), gp_Dir(*pose[:3, 0]))

    if focus == 0:
        projector = HLRAlgo_Projector(ax)
    else:
        projector = HLRAlgo_Projector(ax, focus)

    hlr.Projector(projector)
    hlr.Update()

    hlr_shapes = HLRBRep_HLRToShape(hlr)
    return hlr_shapes


def discretize_compound(compound, tol):
    """
    Given a compound of edges
    Return all edges discretized
    """
    return [d3_to_d2(discretize_edge(edge, tol)) for edge in list(TopologyExplorer(compound).edges())]


def d3_to_d2(points_3d):
    return [tuple(p[:2]) for p in points_3d]


# project a list of 3D points
def project_points(points, args):
    location = args.location
    direction = args.direction
    focus = args.focus

    ax = gp_Ax2(gp_Pnt(*location), gp_Dir(*direction))

    if args.pose is not None:
        pose = args.pose
        ax = gp_Ax2(gp_Pnt(*pose[:3, -1]), gp_Dir(*pose[:3, -2]), gp_Dir(*pose[:3, 0]))

    if focus == 0:
        projector = HLRAlgo_Projector(ax)
    else:
        projector = HLRAlgo_Projector(ax, focus)

    projected = [projector.Project(gp_Pnt(*p)) for p in points]

    return projected