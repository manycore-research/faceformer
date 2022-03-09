"""
Filter closer cases
"""
import argparse, os, trimesh, yaml, time, json
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from tqdm.contrib.concurrent import process_map


def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.linalg.norm(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False, maintain_order=True)


def dist_p2p(vertices, verts_i, verts_j):
    dists = cdist(vertices[verts_i], vertices[verts_j])
    return np.mean(np.min(dists, 1))


def dist_p2l(vertices, verts_i, verts_j, EPS=1e-8, MAX_VALUE=10):
    edges = np.vstack((verts_j[:-1], verts_j[1:])).T
    edge_vector = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_length = np.linalg.norm(edge_vector, axis=1, keepdims=True) + EPS
    edge_tangent = edge_vector / edge_length

    # Points x Lines x Dim
    vector = vertices[verts_i, np.newaxis] - vertices[edges[:, 0]][np.newaxis]

    # Points x Lines
    points_prop = np.sum(
        vector * edge_tangent[np.newaxis], axis=-1) / edge_length.reshape(1, -1)
    points_perp = points_prop[..., np.newaxis] * edge_vector - vector

    # p2l dists within 0 < points_prop < 1
    pl_dists = np.linalg.norm(points_perp, axis=-1)
    pl_valid = np.logical_and(0 < points_prop, points_prop < 1)
    pl_dists[np.logical_not(pl_valid)] = MAX_VALUE

    # p2p dists
    pp_dists = cdist(vertices[verts_i], vertices[edges].reshape(-1, 3))
    pp_dists = pp_dists.reshape(-1, len(edges), 2)
    pp_dists = np.min(pp_dists, -1)

    dists = np.minimum(pl_dists, pp_dists)
    return np.mean(np.min(dists, 1))


def load_and_preprocess(name, args):
    if os.path.exists(os.path.join(args.save_root, f'{name}.npy')):
        return

    mesh_path = os.path.join(args.root, 'obj', f'{name}.obj')
    mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)

    # normalize to a unit sphere
    mesh = scale_to_unit_sphere(mesh)

    feat_path = os.path.join(args.root, 'feat', f'{name}.yml')
    with open(feat_path) as file:
        annos = yaml.full_load(file)

    curve_verts = []
    for curve in annos['curves']:
        vert_indices = np.array(curve['vert_indices']).reshape(-1)
        curve_verts.append(vert_indices)

    vertices = mesh.vertices.view(np.ndarray)
    
    num_curves = len(curve_verts)

    with open(os.path.join(args.save_root, f'{name}.npy'), 'wb') as f:
        np.save(f, vertices)
        np.save(f, num_curves)
        for c in curve_verts:
            np.save(f, c)

def filter_by_thickness(name, args):

    with open(os.path.join(args.save_root, f'{name}.npy'), 'rb') as f:
        vertices = np.load(f)
        num_curves = np.load(f)
        curve_verts = []
        max_index = 0
        for i in range(num_curves):
            curve_verts.append(np.load(f))
            max_index = max(curve_verts[-1].max(), max_index)

    if max_index >= len(vertices):
        print(f"{name} has vertices don't match {len(vertices)} <= {max_index}")
        return None

    # dists = np.zeros((num_curves, num_curves))

    for i in range(num_curves):
        verts_i = curve_verts[i]

        for j in range(i+1, num_curves):
            verts_j = curve_verts[j]

            if args.p2p:
                dist_1 = dist_p2p(vertices, verts_i, verts_j)
                dist_2 = dist_p2p(vertices, verts_j, verts_i)
            else:
                dist_1 = dist_p2l(vertices, verts_i, verts_j)
                dist_2 = dist_p2l(vertices, verts_j, verts_i)

            if dist_1 < args.threshold and dist_2 < args.threshold:
                return None
            # dists[i, j] = dist_1
            # dists[j, i] = dist_2
    return name

def main(args):
    with open("dataset/dataset_gen_logs/filtered_id_list.json", 'r') as f:
        names = json.load(f)

    # preprocess
    process_map(
        partial(load_and_preprocess, args=args), names,
        max_workers=args.num_cores, chunksize=args.num_chunks)

    rets = process_map(
        partial(filter_by_thickness, args=args), names,
        max_workers=args.num_cores, chunksize=args.num_chunks)

    filtered = [ret for ret in rets if ret is not None]

    # Filtering by thickness can take a long time.
    # Uncomment the following and lines in filter_by_thickness() to save intermediate result.
    
    # name_to_dist = {}
    # for i in rets:
    #     if i is None:
    #         continue
    #     name, dists = i
    #     name_to_dist[name] = dists.tolist()
    
    # with open('all_thickness.json', 'w') as f:
    #     json.dump(name_to_dist, f)

    with open('filtered_id_list.json', 'w') as f:
        json.dump(filtered, f)
    
    with open('data_processing_log.txt', 'a') as f:
        f.write("Thickness id list generation done    - " + time.ctime() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/root/Datasets/FaceFormer',
                        help='dataset root')
    parser.add_argument('--save_root', type=str, default='/root/data/curve_verts',
                        help='dataset root')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='threshold for closer edge')
    parser.add_argument('--num_cores', type=int,
                        default=10, help='number of processors.')
    parser.add_argument('--num_chunks', type=int,
                        default=10, help='number of chunk.')
    parser.add_argument('--p2p', action='store_true')
    args = parser.parse_args()

    main(args)
    
