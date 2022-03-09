import argparse
import json
import os
from functools import partial

import numpy as np
import trimesh
import yaml
from tqdm.contrib.concurrent import process_map


def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.linalg.norm(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False, maintain_order=True)


def filter_by_raidus(name, args):
    mesh_path = os.path.join(args.root, 'obj', f'{name}.obj')
    mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    scale = np.linalg.norm(mesh.bounding_box.extents)

    feat_path = os.path.join(args.root, 'feat', f'{name}.yml')
    with open(feat_path) as file:
        annos = yaml.full_load(file)

    radius_array = []

    for curve in annos['curves']:
        # finder the case with thinner cylinder
        if curve['type'] in ['Circle']:
            radius = curve['radius'] / scale

        elif curve['type'] in ['Ellipse']:
            radius = min(curve['maj_radius'], curve['min_radius']) / scale

        else:
            continue

        radius_array.append(radius)

    if len(radius_array) != 0:
        with open(os.path.join(args.root, 'radius', f'{name}.json'), 'w') as f:
            json.dump(min(radius_array), f)

    return name


def main(args):
    with open(os.path.join(args.root, "meta", "filtered_thickness.json"), 'r') as f:
        names = json.load(f)

    os.makedirs(os.path.join(args.root, 'radius'), exist_ok=True)

    # preprocess
    rets = process_map(
        partial(filter_by_raidus, args=args), names,
        max_workers=args.num_cores, chunksize=args.num_chunks)

    filtered = [ret for ret in rets if ret is not None]

    with open(os.path.join(args.root, 'meta', 'filtered_thinness.json'), 'w') as f:
        json.dump(filtered, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/root/Datasets/Faceformer',
                        help='dataset root')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='threshold for closer edge')
    parser.add_argument('--num_cores', type=int,
                        default=8, help='number of processors.')
    parser.add_argument('--num_chunks', type=int,
                        default=64, help='number of chunk.')
    args = parser.parse_args()

    main(args)
