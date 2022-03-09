"""
Render three-view line drawings
"""
import argparse
import os
from functools import partial
import numpy as np
import svgwrite
from cairosvg import svg2png
from tqdm.contrib.concurrent import process_map

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCC.Extend.TopologyUtils import TopologyExplorer

from dataset.utils.json_to_svg import save_png, save_svg
from dataset.utils.read_step_file import read_step_file
from dataset.utils.projection_utils import project_shapes, discretize_compound

O = gp_Pnt(0,0,0)
X = gp_Dir(1,0,0)
Y = gp_Dir(0,1,0)
nY = gp_Dir(0,-1,0)
Z = gp_Dir(0,0,1)

directions = [
    gp_Ax2(O, gp_Dir(1,1,1)), # 45 degree
    gp_Ax2(O, nY, X), # front
    gp_Ax2(O, X, Y), # right
    gp_Ax2(O, Z, X) # top
]

def get_boundingbox(shape, tol=1e-6, use_mesh=False):
    """ return the bounding box of the TopoDS_Shape `shape`
    Parameters
    ----------
    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from
    tol: float
        tolerance of the computed boundingbox
    use_mesh : bool
        a flag that tells whether or not the shape has first to be meshed before the bbox
        computation. This produces more accurate results
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    if use_mesh:
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        if not mesh.IsDone():
            raise AssertionError("Mesh not done.")
    brepbndlib_Add(shape, bbox, use_mesh)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    center = (xmax + xmin) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    extent = abs(xmax-xmin), abs(ymax-ymin), abs(zmax-zmin)
    return center, extent

def get_discretized_edges(name, shape, direction, args):
    """ Given a TopologyExplorer topo, and a face on it,
        find all edges of the face without sewn edges.
        Return discretized edges

    VComponent / HComponent: sharp edges
    Rg1LineVCompound / Rg1LineHCompound: smooth edges
    RgNLineVCompound / RgNLineHCompound: sewn edges
    OutLineVCompound / OutLineHCompound: outlines
    """
    # project the face
    hlr_shapes = project_shapes(shape, direction)
    
    discretized_edges = []

    outline_compound = hlr_shapes.OutLineVCompound()
    if outline_compound:
        discretized_edges += discretize_compound(outline_compound, args.tol)

    smooth_compound = hlr_shapes.Rg1LineVCompound()
    if smooth_compound:
        discretized_edges += discretize_compound(smooth_compound, args.tol)

     # project sharp edges from the face, using only edges. 
    # (to avoid slicing effects from sewn edge when projecting using face)
    sharp_edges_3d = list(TopologyExplorer(shape).edges())
    sharp_edges_compound = project_shapes(sharp_edges_3d, direction).VCompound()
    if sharp_edges_compound:
        sharp_edges_discretized = discretize_compound(sharp_edges_compound, args.tol)

        # check if there are sewn edges
        sewn_compound = hlr_shapes.RgNLineVCompound()
        if sewn_compound: 
            sewn_edges_discretized = discretize_compound(sewn_compound, args.tol)
            for sewn_edge in sewn_edges_discretized:
                try:
                    sharp_edges_discretized.remove(sewn_edge)
                except ValueError:
                    print("sewn edge assumption broken", name)
                    break
        discretized_edges += sharp_edges_discretized
    
    return discretized_edges

def discretized_edge_to_svg_polyline(points):
    """ Returns a svgwrite.Path for the edge, and the 2d bounding box
    """
    return svgwrite.shapes.Polyline(points, fill="none", class_='vectorEffectClass')

def shape_to_svg(shape, name, args):
    """ export a single shape to an svg file and json.
    shape: the TopoDS_Shape to export
    """
    if shape.IsNull():
        raise AssertionError("shape is Null")
    
    for i, direction in enumerate(directions):

        shape_discretized_edges = get_discretized_edges(name, shape, direction, args)
        
        save_svg(shape_discretized_edges, os.path.join(args.root, '3view_svg', f'{name}-{i}.svg'), args)
        
        svg2png(
            bytestring=open(os.path.join(args.root, '3view_svg', f'{name}-{i}.svg'), 'rb').read(),
            output_width=args.width,
            output_height=args.height,
            background_color='white',
            write_to=os.path.join(args.root, '3view_png', f'{name}-{i}.png')
        )

def render_3views(name, args):
    try:
        step_path = os.path.join(args.root, 'step', f'{name}.step')
        # step read timeout at 5 seconds
        try:
            shape, _ = read_step_file(step_path, verbosity=False)
        except:
            print(f"{name} took too long to read")
            return

        if shape is None:
            print(f"{name} is NULL shape")
            return
        
        center, extent = get_boundingbox(shape)

        trans, scale = gp_Trsf(), gp_Trsf()
        trans.SetTranslation(-gp_Vec(*center))
        scale.SetScale(gp_Pnt(0, 0, 0), 2 / np.linalg.norm(extent))
        brep_trans = BRepBuilderAPI_Transform(shape, scale * trans)
        shape = brep_trans.Shape()

        shape_to_svg(shape, name, args)
    except Exception as e:
        print(f"{name} received unknown error", e)

def main(args):
    # all step files
    names = []
    for name in sorted(os.listdir(os.path.join(args.root, 'stat'))):
        names.append(name[:8])

    os.makedirs(os.path.join(args.root, '3view_svg'), exist_ok=True)
    os.makedirs(os.path.join(args.root, '3view_png'), exist_ok=True)

    process_map(
        partial(render_3views, args=args), names,
        max_workers=args.num_cores, chunksize=args.num_chunks
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data",
                        help='dataset root.')
    parser.add_argument('--name', type=str, default=None,
                        help='filename.')
    parser.add_argument('--num_cores', type=int,
                        default=40, help='number of processors.')
    parser.add_argument('--num_chunks', type=int,
                        default=10, help='number of chunk.')
    parser.add_argument('--width', type=int,
                        default=256, help='svg width.')
    parser.add_argument('--height', type=int,
                        default=256, help='svg height.')
    parser.add_argument('--tol', type=float,
                        default=1e-4, help='svg discretization tolerance.')
    parser.add_argument('--line_width', type=str,
                        default=str(3/256), help='svg line width.')
    parser.add_argument('--filter_num_shapes', type=int,
                        default=8, help='do not process step files \
                            that have more than this number of shapes.')
    parser.add_argument('--filter_num_edges', type=int,
                        default=1000, help='do not process step files \
                            that have more than this number of edges.')

    args = parser.parse_args()

    if args.name is None:
        main(args)
    else:
        render_3views(args.name, args)
