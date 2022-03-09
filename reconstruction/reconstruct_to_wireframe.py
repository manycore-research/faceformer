import argparse
import itertools
import json
import os
from functools import partial

import cvxpy as cp
import numpy as np
import open3d as o3d
from surfaceformer.utils import flatten_list
from tqdm.contrib.concurrent import process_map

from reconstruction.reconstruction_utils import (construct_connected_cylinder,
                                                 dist, fit_curve,
                                                 is_straight_line)

INTERMEDIATE_TYPE = 11 # the 4 extra faces added to each cylinder, not used in final reconstruction

def sample_points_on_line(line, sample_dist):
    x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    num_samples = int(np.sqrt((x1-x2)**2+(y1-y2)**2) / sample_dist) + 1
    t = np.linspace(0, 1, num_samples)
    x = x1 + (x2-x1) * t
    y = y1 + (y2-y1) * t
    return np.vstack([x, y]).T

def reconstruct_file(name, root):
    try:
        if os.path.exists(os.path.join(root, 'ply', f"{name}.ply")):
            return
        data = json.load(open(os.path.join(root, 'json', f'{name}.json')))
        num_faces = len(data['pred_faces'])
        num_edges = len(data['edges'])

        to_add_new_planes = []
        to_add_new_edges = []
        face_removal_indices = []
        circle_face_to_construct = [] # should contain indices of two outlines
        circle_face_to_construct_dir = [] # should contain the direction of the outlines

        dom_directions = [np.array(d[:2]) / np.linalg.norm(d[:2]) for d in data['dominant_directions']]
        face_to_normal = {}

        # check face types for other
        for i, (face_type, loops) in enumerate(data['pred_faces']):
            if face_type not in [0, 1]:
                face_removal_indices.append(i)
                continue

            # for each cylinder face, we construct two new plane face
            # always select the mid point for plane reconstruction
            if face_type == 1:
                face_removal_indices.append(i)

                # cylinder face should have two curves and two straight lines
                all_edge_inds = list(loops)
                all_edges = [data['edges'][i] for i in all_edge_inds]
                # if more than two straight lines, not a cylinder, skip
                count = 0
                for edge in all_edges:
                    if is_straight_line(edge):
                        count += 1
                if count != 2:
                    print(f"{name} has {count} straight lines, not a cylinder")
                    continue
                try:
                    all_edges, all_edge_inds, all_dirs = construct_connected_cylinder(all_edges, all_edge_inds)
                except:
                    continue
                
                # assuming loop has 4 edges as a cylinder face
                if len(all_edges) != 4:
                    # combine nearing curves
                    i = 0
                    while i < len(all_edges):
                        next_edge_ind = (i+1) % len(all_edges)
                        if not is_straight_line(all_edges[i]) and not is_straight_line(all_edges[next_edge_ind]):
                            all_edges[i] += all_edges[next_edge_ind]
                            all_edges.pop(next_edge_ind)
                            all_edge_inds.pop(next_edge_ind)
                            all_dirs.pop(next_edge_ind)
                            continue
                        i += 1
                    if len(all_edges) != 4:
                        print(f"{name} has {len(all_edges)} edges in a cylinder")
                        continue

                # assert face is a wireframe with coedge directions
                # if straight line comes first, 
                # then direction of new constructed line is opposite of the straight line
                if is_straight_line(all_edges[0]):
                    line_ind = all_edge_inds[0]
                    line = all_edges[0]
                    line_dir = all_dirs[0]
                    curve = all_edges[1]
                    curve_ind = all_edge_inds[1]
                    other_line_ind = all_edge_inds[2]
                    other_line = all_edges[2]
                    other_line_dir = all_dirs[2]
                    other_curve_ind = all_edge_inds[3]
                else:
                    curve = all_edges[0]
                    curve_ind = all_edge_inds[0]
                    other_line = all_edges[1]
                    other_line_ind = all_edge_inds[1]
                    other_line_dir = all_dirs[1]
                    other_curve_ind = all_edge_inds[2]
                    line = all_edges[3]
                    line_ind = all_edge_inds[3]
                    line_dir = all_dirs[3]
        
                # assert all length of cylinder straight lines are the same 
                # displace midpoint of one curve the same amount to generate the middle edge

                direction = np.array(line[0]) - np.array(line[1])
                mid_point = np.array(curve[len(curve) // 2])
                # next_point = np.array(other_curve[len(other_curve) // 2])

                next_point = mid_point + direction
                mid_point = mid_point.tolist()
                next_point = next_point.tolist()
                new_mid_edge = [mid_point, next_point]
                new_edges = [new_mid_edge, [line[0], next_point], [line[1], mid_point], [other_line[1], next_point], [other_line[0], mid_point]]
                ind_offset = len(to_add_new_edges) + num_edges
                to_add_new_edges += new_edges
                face_1 = (INTERMEDIATE_TYPE, [line_ind, 2+ind_offset, ind_offset, 1+ind_offset])
                face_2 = (INTERMEDIATE_TYPE, [other_line_ind, 3+ind_offset, ind_offset, 4+ind_offset])
                to_add_new_planes += [face_1, face_2]
                circle_face_to_construct.append([line_ind, other_line_ind, ind_offset, curve_ind, other_curve_ind])
                circle_face_to_construct_dir.append([line_dir, other_line_dir, 1])

                # find plane's normal, assuming the axis of the cylinder face is aligned with one of the dominant directions
                edge_direction = np.array(line[0]) - np.array(line[1])
                normal_direction_ind = np.argmax([np.abs(np.dot(edge_direction, d)) for d in dom_directions])

                # find other coedge's circle plane and add normal constraint
                for i, (face_type, indices) in enumerate(data['pred_faces']):
                    if curve_ind in indices or other_curve_ind in indices:
                        face_to_normal[tuple(indices)] = normal_direction_ind
                
        # Add in new faces
        data['pred_faces'] += to_add_new_planes
        data['edges'] += to_add_new_edges
        num_faces = len(data['pred_faces'])
        num_edges = len(data['edges'])

        
        
        removed_faces = []
        # remove cylinder faces
        for i, ind in enumerate(face_removal_indices):
            removed_faces.append(data['pred_faces'].pop(ind-i))

        P = []
        b = []
        C = []

        
        # create equation for each perpendicular face's normal direction and dominant direction
        # check 2D parallelism to one of the dominant directions
        # edge parallel to one dominant direction => perpendicular to that direction

        # normalized dom_directions
        # # !! 2d dominant directions need human input
        # aa = -np.sum(dom_directions[0]*dom_directions[1])
        # bb = -np.sum(dom_directions[1]*dom_directions[2])
        # cc = -np.sum(dom_directions[0]*dom_directions[2])
        # z3 = np.sqrt(cc*bb / aa)
        # z2 = bb / z3
        # z1 = cc / z3

        # origin_directions = [dom_directions[0].tolist()+[z1], dom_directions[1].tolist()+[z2], dom_directions[2].tolist()+[z3]]
        # origin_directions = [np.array(d) / np.linalg.norm(d) for d in origin_directions]
        origin_directions = [np.array(d) / np.linalg.norm(d) for d in data['dominant_directions']]
        face_removal_indices = []
        for face_ind, (face_type, indices) in enumerate(data['pred_faces']):
            parallel_count_for_dom_directions = [0] * 3
            for edge_ind in indices:
                edge = data['edges'][edge_ind]
                if not is_straight_line(edge):
                    continue
                edge_direction = np.array(edge[0]) - np.array(edge[1])
                edge_direction /= np.linalg.norm(edge_direction)
                # check if edge is parallel to one of the dominant directions

                for i, direction in enumerate(dom_directions):
                    if np.abs(np.dot(edge_direction, direction)) > (1 - 1e-10):
                        parallel_count_for_dom_directions[i] += 1
            

            # cylinder planes have predetermined normals from the outline
            if tuple(indices) in face_to_normal:
                normal_ind = face_to_normal[tuple(indices)]
                for i in range(3):
                    if i != normal_ind:
                        parallel_count_for_dom_directions[i] += 1

            if 0 not in parallel_count_for_dom_directions:
                # parallel to all dominant directions => wrong face prediction
                face_removal_indices.append(face_ind)
                continue

            # perpendicular to parallel directions
            for ind, count in enumerate(parallel_count_for_dom_directions):
                if count != 0:
                    row = np.zeros(3 * num_faces)
                    direction_3d = origin_directions[ind]
                    # account for face removal
                    face_ind -= len(face_removal_indices)
                    row[3*face_ind: 3*face_ind+2] = [direction_3d[0], direction_3d[1]]
                    brow = np.array([direction_3d[2]])
                    P.append(row)
                    b.append(brow)
        for i, ind in enumerate(face_removal_indices):
            data['pred_faces'].pop(ind-i)    
                
        # find all unique vertices
        all_vertices = []
        all_used_edges = set(flatten_list([indices for _, indices in data['pred_faces']]))
        for ind in all_used_edges:
            all_vertices += data['edges'][ind]

        unique_vertices = []
        tol = 1e-4
        for vertex in all_vertices:
            dists = np.array([dist(p1, vertex) for p1 in unique_vertices])
            if np.sum(dists < tol) < 1:
                # new unique vertex
                unique_vertices.append(vertex)

        face_grouped_by_vertex = [[] for _ in range(len(unique_vertices))]
        # match faces to vertices
        for face_ind, (_, indices) in enumerate(data['pred_faces']):
            for edge_ind in indices:
                for point in data['edges'][edge_ind]:
                    dists = np.array([dist(p1, point) for p1 in unique_vertices])
                    vertex_ind = np.argmin(dists)
                    face_grouped_by_vertex[vertex_ind].append(face_ind)

        face_grouped_by_vertex = [list(set(group)) for group in face_grouped_by_vertex]
        for vertex, face_group in zip(unique_vertices, face_grouped_by_vertex):
            if len(face_group) < 2:
                continue
            # for each two face joined on 1 vertex, we create one equation for them
            for f1, f2 in itertools.combinations(face_group, 2):
                row = np.zeros(3 * num_faces)
                row[f1*3: f1*3 + 3] = [vertex[0], vertex[1], 1]
                row[f2*3: f2*3 + 3] = [-vertex[0], -vertex[1], -1]
                brow = np.array([0])
                P.append(row)
                b.append(brow)
            # for each vertex and face, we create one constraint that z > 0
            for f in face_group:
                row = np.zeros(3 * num_faces)
                row[f*3: f*3 + 3] = [-vertex[0], -vertex[1], -1]
                C.append(row)

        P = np.array(P)
        C = np.array(C)
        b = np.array(b)

        n = P.shape[-1]
        if n == 0:
            return
        if C.shape[-1] == 0:
            return

        # sample points for 3d reconstruction
        pts = []
        pts_label = []
        sample_dist = 5e-3
        ind_to_3d_map = {} # contains the start of 3d samples for each edge index
        mid_edge_to_remove_start_ind = []
        mid_edge_inds = []
        for face_ind, (face_type, indices) in enumerate(data['pred_faces']):
            if face_type == INTERMEDIATE_TYPE:
                # only the first line (outline) and the third (mid edge) of intermediate face needs to be reconstructed
                sampled_pts = sample_points_on_line(data['edges'][indices[0]], sample_dist)
                pts.append(sampled_pts)
                ind_to_3d_map[indices[0]] = (len(pts_label), len(sampled_pts))
                pts_label += [face_ind] * len(sampled_pts)
                # memorize where the outline's corresponding 3d points are
                sampled_pts = sample_points_on_line(data['edges'][indices[2]], sample_dist)
                pts.append(sampled_pts)
                ind_to_3d_map[indices[2]] = (len(pts_label), len(sampled_pts))
                mid_edge_to_remove_start_ind.append(len(pts_label))
                mid_edge_inds.append(indices[2])
                pts_label += [face_ind] * len(sampled_pts)
                continue
            for edge_ind in indices:
                if is_straight_line(data['edges'][edge_ind]):
                    sampled_pts = sample_points_on_line(data['edges'][edge_ind], sample_dist)
                    pts.append(sampled_pts)
                    ind_to_3d_map[edge_ind] = (len(pts_label), len(sampled_pts))
                    pts_label += [face_ind] * len(sampled_pts)

        if len(pts) == 0:
            return
        pts = np.vstack(pts)
        pts_label = np.array(pts_label)
        
        f = cp.Variable((n, 1))
        try:
            objective = cp.Minimize(cp.norm1(P @ f + b))
            constraints = [C @ f >= 0]
            prob = cp.Problem(objective, constraints)

            result = prob.solve()
        except:
            return
        params = f.value.reshape(-1, 3)

        N = len(pts)

        pts_one = np.hstack((pts, np.ones((N, 1))))

        depth = np.sum(params[pts_label] * pts_one, axis=1, keepdims=True)

        xyz = np.hstack((pts, depth))

        # reconstruct the circle planes
        for i in range(len(circle_face_to_construct)):
            line_ind, other_line_ind, mid_edge_ind, curve_ind, other_curve_ind = circle_face_to_construct[i]
            line_dir, other_line_dir, mid_edge_dir = circle_face_to_construct_dir[i]
            # connections between two outlines give our center of the circle
            # direction of outline give us normal of the plane
            start_ind, num_samples = ind_to_3d_map[line_ind]
            pts = xyz[start_ind:start_ind+num_samples]
            other_start_ind, num_samples = ind_to_3d_map[other_line_ind]
            other_pts = xyz[other_start_ind:other_start_ind+num_samples]
            mid_edge_start_ind, num_samples = ind_to_3d_map[mid_edge_ind]
            mid_edge_pts = xyz[mid_edge_start_ind:mid_edge_start_ind+num_samples]

            p1, p2, p3 = pts[::line_dir][0], other_pts[::other_line_dir][-1], mid_edge_pts[::mid_edge_dir][-1]
            curve_pts = fit_curve(p1, p2, p3)
            ind_to_3d_map[other_curve_ind] = (len(xyz), len(curve_pts))
            xyz = np.vstack([xyz, curve_pts])
            
            p1, p2, p3 = pts[::line_dir][-1], other_pts[::other_line_dir][0], mid_edge_pts[::mid_edge_dir][0]
            curve_pts = fit_curve(p1, p2, p3)
            ind_to_3d_map[curve_ind] = (len(xyz), len(curve_pts))
            xyz = np.vstack([xyz, curve_pts])

        
        # add back the removed cylinder faces
        data['pred_faces'] += removed_faces

        # iterate through the faces and add edges that are not mid-edges
        points = []
        edges_drawn = set(mid_edge_inds)
        for face_type, indices in data['pred_faces']:
            if face_type == INTERMEDIATE_TYPE:
                continue
            for ind in indices:
                if ind in ind_to_3d_map and ind not in edges_drawn:
                    start_ind, length = ind_to_3d_map[ind]
                    points.append(xyz[start_ind:start_ind+length])
                    edges_drawn.add(ind)

        pcd = o3d.geometry.PointCloud()
        pts = np.vstack(points)
        pts[:, 1] = -pts[:, 1]
        pcd.points = o3d.utility.Vector3dVector(pts)

        o3d.io.write_point_cloud(os.path.join(root, 'ply', f"{name}.ply"), pcd)
    except:
        print(f"{name} failed")
        return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="/root/data",
                        help='dataset root.')
    parser.add_argument('--name', type=str, default=None,
                        help='filename.')
    parser.add_argument('--num_cores', type=int,
                        default=10, help='number of processors.')
    parser.add_argument('--num_chunks', type=int,
                        default=5, help='number of chunk.')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.root, 'ply'), exist_ok=True)

    if args.name is not None:
        reconstruct_file(args.name, args.root)
    else:
        all_names = [name[:8] for name in os.listdir(os.path.join(args.root, 'json'))]
        process_map(partial(reconstruct_file, root=args.root), all_names, 
            max_workers=args.num_cores, chunksize=args.num_chunks)
        # for name in all_names:
        #     reconstruct_file(name, args.root)
