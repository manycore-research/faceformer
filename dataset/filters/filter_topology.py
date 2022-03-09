"""
Topology Filtering
* Group all objects of similar topology in the same bin
"""

import argparse
import json
import os

import yaml
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

types_of_curves = {
    "Line": 0, "Circle": 1, "Ellipse": 2, "BSpline": 3, "Other": 4}
types_of_surfs = {
    "Plane": 0, "Cylinder": 1, "Cone": 2, "Sphere": 3, "Torus": 4,
    "Revolution": 5, "Extrusion": 6, "BSpline": 7, "Other": 8}


def main(args):
    # all step files
    names = []
    for name in sorted(os.listdir(os.path.join(args.root, 'stat'))):
        names.append(name[:8])

    if os.path.exists(args.error_log):
        # remove shapes that give errors
        with open(args.error_log, 'r') as f:
            lines = f.read().splitlines()

        error_names = [line[:8] for line in lines if line[:8].isdigit()]
        id_list = [name for name in names if name not in set(error_names)]
    else:
        print("error log not found")
        id_list = names

    # gather topology info for each object as their feature
    names = []
    features = []
    for name in tqdm(id_list):
        names.append(name)
        path = os.path.join(args.root, 'stat', f'{name}.yml')
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        curves = [types_of_curves[curve] for curve in data['curves']]
        surfs = [types_of_surfs[surf] for surf in data['surfs']]

        curves_hist = [0] * len(types_of_curves)
        for curve in curves:
            curves_hist[curve] += 1
        surfs_hist = [0] * len(types_of_surfs)
        for surf in surfs:
            surfs_hist[surf] += 1

        feature = [data['#edges'], data['#parts'], data['#sharp'],
                   data['#surfs'], *curves_hist, *surfs_hist]
        features.append(feature)

    # Use nearest neighbors to find clusterings
    neigh = NearestNeighbors()
    neigh.fit(features)
    dist, indices = neigh.radius_neighbors(features, args.similarity_threshold)
    bins = set([tuple(ind) for ind in indices])
    list_names = []
    for b in bins:
        name_bin = [names[i] for i in b]
        list_names.append(name_bin)
    with open("dataset/dataset_gen_logs/topo_matching_bins.json", 'w') as f:
        json.dump(list_names, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data",
                        help='dataset root.')
    parser.add_argument('--error_log', type=str,
                        default="dataset/dataset_gen_logs/error.txt",
                        help='dataset generation error log.')
    parser.add_argument('--similarity_threshold', type=float,
                        default=0,
                        help="grouping threhold for similarity")
    args = parser.parse_args()

    main(args)
