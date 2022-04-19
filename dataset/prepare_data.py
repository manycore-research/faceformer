"""
Data Prep for ABC step Files
* use HLR to render wireframe
* break surfaces with outlines
* remove seam lines
"""
import argparse
import json
import os
from functools import partial

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Extend.TopologyUtils import TopologyExplorer
from tqdm.contrib.concurrent import process_map

from dataset.utils.discretize_edge import (DiscretizedEdge, sort_faces_by_indices, 
                                            sort_edges_by_coordinate)
from dataset.utils.json_to_svg import save_png, save_svg, save_svg_groups
from dataset.utils.read_step_file import read_step_file
from dataset.utils.TopoMapper import TopoMapper

from dataset.tests.check_faces_enclosed import is_face_enclosed
from faceformer.utils import flatten_list
from dataset.utils.projection_utils import generate_random_camera_pos

def get_boundingbox(shapes, tol=1e-6):
    """ return the bounding box of the TopoDS_Shape `shape`
    Parameters
    ----------
    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from
    tol: float
        tolerance of the computed boundingbox
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    for shape in shapes:
        brepbndlib_Add(shape, bbox, False)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    center = (xmax + xmin) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    extent = abs(xmax-xmin), abs(ymax-ymin), abs(zmax-zmin)
    return center, extent


def shape_to_svg(shape, name, args):
    """ export a single shape to an svg file and json.
    shape: the TopoDS_Shape to export
    """
    if shape.IsNull():
        raise AssertionError("shape is Null")

    topo = TopoMapper(shape, args)
    all_dedges = []
    faces_pointers = []
    face_types = []
    # face_parameters = []
    all_shrinked_dedges = []
    shape_center, _ = get_boundingbox([shape])
    # ind = 2
    # replacements = {2:[ind, (3, (ind-2)%3)], 3:[(ind-2)%3, (2, ind)]}

    for index, face in enumerate(topo.all_faces.values()):
        discretized_edges = face.get_oriented_dedges()
        discretized_edges_3d = face.get_oriented_dedges(is_3d=True)

        # generate smaller edges for visualization of faces
        face_edges_lists = [edge.edges for edge in face.edges]
        face_edges = flatten_list(face_edges_lists)
        center, _ = get_boundingbox(face_edges)
        translation = gp_Trsf()
        push_vec = np.array([center[0] - shape_center[0], center[1] - shape_center[1], center[2] - shape_center[2]]) * 1.04
        # push_vec += push_vec / np.sqrt(np.sum(push_vec**2)) * 0.1 # push a fixed amount
        # push the edges out along the line from the center to the face
        translation.SetTranslation(gp_Vec(*push_vec))
        
        # scale = gp_Trsf()
        # scale.SetScale(gp_Pnt(*center), 0.7)
        shrinked_dedges = []
        for i, edge_list in enumerate(face_edges_lists):
            # if index in replacements and i == replacements[index][0]:
            #     other_face_index = replacements[index][1][0]
            #     other_face_edges_lists = [edge.edges for edge in list(topo.all_faces.values())[other_face_index].edges]
            #     edge_list = other_face_edges_lists[replacements[index][1][1]]
            shrinked_edges = []
            for edge in edge_list:
                brep_trans = BRepBuilderAPI_Transform(edge, translation)
                edge = brep_trans.Shape()
                shrinked_edges.append(edge)
            shrinked_dedges.append(topo._raw_project(shrinked_edges, args.discretize_last))

        all_shrinked_dedges.append(shrinked_dedges)
        filename = os.path.join(args.root, 'face_svg', f'{name}_{index}.svg')
        save_svg(discretized_edges, filename, args)
        save_png(f'{name}_{index}', args, prefix='face_')

        
        filename = os.path.join(args.root, 'face_shrinked_face_svg', f'{name}_{index}.svg')
        save_svg(shrinked_dedges, filename, args)
        save_png(f'{name}_{index}', args, prefix='face_shrinked_face_')

        # generate data for face ground truth
        if args.combine_coedge:
            # combine coedges => all coedges share the same direction
            for edge in face.edges:
                if edge.DiscretizedEdge is None:
                    edge.DiscretizedEdge = DiscretizedEdge(edge.dedge)
                    all_dedges.append(edge.DiscretizedEdge)
            face_pointers = [edge.DiscretizedEdge for edge in face.edges]
            faces_pointers.append(face_pointers)
        else:
            assert len(discretized_edges) == len(shrinked_dedges)
            assert len(discretized_edges) == len(discretized_edges_3d)
            # each edge is represented as two discretized edges in two directions
            # save 3d points
            face_pointers = [DiscretizedEdge(dedge, smaller_edge=shrinked_dedge, edge3d=dedge_3d) \
                                for dedge, shrinked_dedge, dedge_3d in zip(discretized_edges, shrinked_dedges, discretized_edges_3d)]
            all_dedges += face_pointers
            # get face pointers
            faces_pointers.append(face_pointers)
        
        face_types.append(face.face_type)
        # face_parameters.append(face.parameters)
    
    all_dedges = sort_edges_by_coordinate(all_dedges)
    # assign index to each dedge
    for index, dedge in enumerate(all_dedges):
        dedge.index = index
    
    faces_indices = []
    for face_pointers in faces_pointers:
        if args.order_by_position:
            faces_indices.append(sorted([dedge.index for dedge in face_pointers]))
        else:
            faces_indices.append([dedge.index for dedge in face_pointers])
    
    save_svg([edge.dedge for edge in topo.all_edges.values()], os.path.join(
        args.root, 'svg', f'{name}.svg'), args)
    save_png(name, args)
    save_svg_groups(all_shrinked_dedges, os.path.join(
        args.root, 'face_shrinked_svg', f'{name}.svg'), args)
    save_png(name, args, prefix='face_shrinked_')

    if args.combine_coedge:
        faces_indices = [np.roll(face, -np.argmin(face), axis=0).tolist() for face in faces_indices]
        faces_indices = sort_faces_by_indices(faces_indices)
    else:
        # check enclosedness here, raise error if not enclosed
        # group indices
        all_edge_points = [dedge.points for dedge in all_dedges]
        sorted_faces_indices = []
        for i, face in enumerate(faces_indices):
            all_face_loops = is_face_enclosed(all_edge_points, face, args.tol * 2)
            if not all_face_loops:
                raise Exception("faces unenclosed")
            # roll enclosed loops so smallest index is at the front
            all_face_loops = [np.roll(loop, -np.argmin(loop), axis=0).tolist() for loop in all_face_loops]
            # loops are ordered by first index
            all_face_loops = sorted(all_face_loops, key=lambda x: x[0])
            # sorted_faces_indices.append([face_types[i], all_face_loops, face_parameters[i]])
            if args.no_face_type:
                sorted_faces_indices.append(all_face_loops)
            else:
                sorted_faces_indices.append([face_types[i], all_face_loops])

        # each face: 
        # [
        #   type, 
        #   [loops],
        #   [parameters]            
        # ]
        # order faces by first index
        if args.no_face_type:
            faces_indices = sorted(sorted_faces_indices, key=lambda x: x[0][0])
        else:
            faces_indices = sorted(sorted_faces_indices, key=lambda x: x[1][0][0])
    edges_to_json(all_dedges, faces_indices, name, topo.get_dominant_directions())


def shape_to_svg_direction_token(shape, name, args):
    """ export a single shape to an svg file and json.
    ! combine coedge, and for each face index, give a direction indicator 0/1
    shape: the TopoDS_Shape to export
    """
    if shape.IsNull():
        raise AssertionError("shape is Null")

    topo = TopoMapper(shape, args)
    all_dedges = []
    faces_pointers = []
    face_types = []

    for index, face in enumerate(topo.all_faces.values()):
        # save shape visualization
        discretized_edges = face.get_oriented_dedges()
        filename = os.path.join(args.root, 'face_svg', f'{name}_{index}.svg')
        save_svg(discretized_edges, filename, args)
        save_png(f'{name}_{index}', args, prefix='face_')

        # generate data for face ground truth
        for edge in face.edges:
            if edge.DiscretizedEdge is None:
                edge.DiscretizedEdge = DiscretizedEdge(edge.dedge)
                all_dedges.append(edge.DiscretizedEdge)
        # e-> edge, o-> orientation
        face_pointers = [(e.DiscretizedEdge, o) for e, o in zip(face.edges, face.edge_orientations)]
        faces_pointers.append(face_pointers)
        face_types.append(face.face_type)
        
    # save face visualization
    save_svg([edge.dedge for edge in topo.all_edges.values()], os.path.join(
        args.root, 'svg', f'{name}.svg'), args)
    save_png(name, args)

    # generate data for face ground truth
    all_dedges = sort_edges_by_coordinate(all_dedges)
    # assign index to each dedge
    for index, dedge in enumerate(all_dedges):
        dedge.index = index
    
    faces_indices = []
    for face_pointers in faces_pointers:
        # o-> orientation
        faces_indices.append([(dedge.index, o) for dedge, o in face_pointers])

    # check enclosedness here, raise error if not enclosed
    # group indices
    all_edge_points = [dedge.points for dedge in all_dedges]
    sorted_faces_indices = []
    for face in faces_indices:
        all_face_loops = is_face_enclosed(all_edge_points, face, args.tol * 2)
        if not all_face_loops:
            raise Exception("faces unenclosed")
        # roll enclosed loops so smallest index is at the front
        all_face_loops = [np.roll(loop, -np.argmin([t[0] for t in loop]), axis=0).tolist() for loop in all_face_loops]
        # loops are ordered by first index
        all_face_loops = sorted(all_face_loops, key=lambda x: x[0][0])
        sorted_faces_indices.append(all_face_loops)

    # order faces by first index
    faces_indices = sorted(sorted_faces_indices, key=lambda x: x[0][0][0])
    edges_to_json(all_dedges, faces_indices, name, topo.get_dominant_directions())



def edges_to_json(all_dedges, faces_indices, name, dominant_directions):
    # write to json
    json_filename = os.path.join(args.root, 'json', f'{name}.json')
    data = {}
    data['edges'] = [dedge.points for dedge in all_dedges]
    data['edges3d'] = [dedge.edge3d for dedge in all_dedges]
    data['shrinked_edges'] = [dedge.smaller_edge for dedge in all_dedges]
    data['faces_indices'] = faces_indices
    data['dominant_directions'] = dominant_directions
    data['pairings'] = {}
    # find all pairings of indices
    for i in range(len(data['edges'])):
        for j in range(i+1, len(data['edges'])):
            if data['edges'][i] == data['edges'][j][::-1]:
                data['pairings'][i] = j
    with open(json_filename, 'w') as f:
        json.dump(data, f)


def render_shape_and_faces(name, args):
    try:
        # if os.path.exists(os.path.join(args.root, 'json', f'{name}.json')):
        #     return
        step_path = os.path.join(args.root, 'step', f'{name}.step')
        # step read timeout at 5 seconds
        try:
            shape, num_shapes = read_step_file(step_path, verbosity=False)
        except:
            print(f"{name} took too long to read")
            return

        if shape is None:
            print(f"{name} is NULL shape")
            return

        if num_shapes > args.filter_num_shapes:
            print(f"{name} has {num_shapes} shapes. Too many!")
            return

        topology_explorer = TopologyExplorer(shape)

        if len(list(topology_explorer.edges())) > args.filter_num_edges:
            print(f"{name} has too many edges.")
            return

        center, extent = get_boundingbox([shape])

        trans, scale = gp_Trsf(), gp_Trsf()
        trans.SetTranslation(-gp_Vec(*center))
        scale.SetScale(gp_Pnt(0, 0, 0), 2 / np.linalg.norm(extent))
        brep_trans = BRepBuilderAPI_Transform(shape, scale * trans)
        shape = brep_trans.Shape()
        
        args.pose = None
        # generate random camera position
        if args.random_camera:
            # 5 tries at random angle image
            for _ in range(5):
                try:
                    focus, cam_pose = generate_random_camera_pos(args.seed)
                    args.pose = cam_pose
                    # check orthographic projection
                    if args.focus != 0:
                        args.focus = focus
                    if args.direction_token:
                        shape_to_svg_direction_token(shape, name, args)
                    else:
                        shape_to_svg(shape, name, args)
                    return
                except:
                    continue

        if args.direction_token:
            shape_to_svg_direction_token(shape, name, args)
        else:
            shape_to_svg(shape, name, args)

    except Exception as e:
        print(f"{name} received unknown error", e)

def prepare_splits(args):
    if os.path.exists(args.id_list):
        with open(args.id_list, 'r') as f:
            names = json.load(f)
    else:
        names = []
        for name in sorted(os.listdir(os.path.join(args.root, 'json'))):
            names.append(name[:8])
    
    np.random.seed(args.seed)
    np.random.shuffle(names)
    train_ratio, valid_ratio, test_ratio = args.split
    trainlist, validlist, testlist = np.split(names, [int(
        len(names) * train_ratio), int(len(names) * (train_ratio + valid_ratio))])

    np.savetxt(os.path.join(args.root, 'train.txt'), trainlist, fmt="json/%s.json")
    np.savetxt(os.path.join(args.root, 'valid.txt'), validlist, fmt="json/%s.json")
    np.savetxt(os.path.join(args.root, 'test.txt'), testlist, fmt="json/%s.json")


def main(args):
    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.root, 'svg'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'png'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'face_shrinked_face_svg'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'face_shrinked_face_png'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'face_shrinked_svg'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'face_shrinked_png'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'face_svg'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'face_png'), exist_ok=True)
    os.makedirs(os.path.join(args.root, 'json'), exist_ok=True)
    
    if os.path.exists(args.id_list):
        with open(args.id_list, 'r') as f:
            names = json.load(f)
    else:
        names = []
        for name in sorted(os.listdir(os.path.join(args.root, 'step'))):
            names.append(os.path.splitext(name)[0])

    if not args.only_split:
        process_map(
            partial(render_shape_and_faces, args=args), names,
            max_workers=args.num_cores, chunksize=args.num_chunks
        )

    prepare_splits(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data",
                        help='dataset root.')
    parser.add_argument('--id_list', type=str, default="None",
                        help='filtered(with similarity) data id list')
    parser.add_argument('--name', type=str, default=None,
                        help='filename.')
    parser.add_argument('--num_cores', type=int,
                        default=5, help='number of processors.')
    parser.add_argument('--num_chunks', type=int,
                        default=10, help='number of chunk.')
    parser.add_argument('--width', type=int,
                        default=256, help='svg width.')
    parser.add_argument('--height', type=int,
                        default=256, help='svg height.')
    parser.add_argument('--png_padding', type=float,
                        default=0.2, help='padding from content to the edge of png.')
    parser.add_argument('--tol', type=float,
                        default=1e-4, help='svg discretization tolerance.')
    parser.add_argument('--face_shrink_scale', type=float,
                        default=0.8, help='shrinking face for visualization.')
    parser.add_argument('--line_width', type=str,
                        default=str(6/256), help='svg line width.')
    parser.add_argument('--filter_num_shapes', type=int,
                        default=1, help='do not process step files \
                            that have more than this number of shapes.')
    parser.add_argument('--filter_num_edges', type=int,
                        default=64, help='do not process step files \
                            that have more than this number of edges.')
    parser.add_argument('--location', nargs="+", type=float, 
                        default=[1, 1, 1], help='projection location')
    parser.add_argument('--direction', nargs="+", type=float, 
                        default=[1, 1, 1], help='projection direction')
    parser.add_argument('--focus', type=float,
                        default=3, help='focus of the projection camera.')
    parser.add_argument('--split', nargs="+", type=int, 
                        default=[0.93, 0.02, 0.05],
                        help='train/valid/test split ratio')
    parser.add_argument('--only_split', action='store_true')
    parser.add_argument('--combine_coedge', action='store_true')
    parser.add_argument('--order_by_position', action='store_true')
    parser.add_argument('--direction_token', action='store_true')
    parser.add_argument('--random_camera', action='store_true')
    parser.add_argument('--discretize_last', action='store_true')
    parser.add_argument('--no_face_type', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help='numpy random seed')

    args = parser.parse_args()

    if args.name is None:
        main(args)
    else:
        render_shape_and_faces(args.name, args)
