"""
Three view Filtering
* Further divide each bin after topology filtering
* By the similarity in three-view line drawings
"""

import argparse
import json
import os

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def main(args):
    # topology bins
    with open("dataset/dataset_gen_logs/topo_matching_bins.json", 'r') as f:
        list_names = json.load(f)

    # bins of more than 1 objects
    multi_bins = [b for b in list_names if len(b) > 1]

    # 3view error list
    with open(args.error_log, 'r') as f:
        lines = f.read().splitlines()
        error_names = [line[:8] for line in lines if line[:8].isdigit()]
    error_names = set(error_names)

    # remove error objects from topology bins
    filtered_multi_bins = []
    new_bins = []
    for b in multi_bins:
        new_bin = [name for name in b if name not in error_names]
        if len(new_bin) == 0:
            continue
        if len(new_bin) == 1:
            new_bins.append(new_bin)
        else:
            filtered_multi_bins.append(new_bin)

    # cluster by jaccard distance in three views
    for large_bin in tqdm(filtered_multi_bins):
        all_bin_imgs = []
        for name in large_bin:
            feature = []
            for i in range(1, 4):
                img_path = os.path.join(
                    args.root, '3view_png', f'{name}-{i}.png')
                originalImage = cv2.imread(img_path)
                if originalImage is None:
                    feature.append(np.ones(128*128))
                    continue
                halfImage = cv2.resize(originalImage, (0, 0), fx=0.5, fy=0.5)
                grayImage = cv2.cvtColor(halfImage, cv2.COLOR_BGR2GRAY)
                thresh, bin_img = cv2.threshold(
                    grayImage, 254, 255, cv2.THRESH_BINARY)
                feature.append(bin_img)
            all_bin_imgs.append(np.array(feature).flatten())

        X = np.array(all_bin_imgs) == 0

        dist_mat = pairwise_distances(X, metric='jaccard')
        clusters = AgglomerativeClustering(n_clusters=None, affinity='precomputed', 
                                           distance_threshold=args.similarity_threshold, linkage='single').fit(dist_mat)
        classes = clusters.labels_
        new_bin = [[] for _ in range(max(classes)+1)]
        for name, c in zip(large_bin, classes):
            new_bin[c].append(name)
        new_bins += new_bin

    # add bins of single object back
    for b in list_names:
        if len(b) == 1:
            new_bins.append(b)

    # generate a list of valid, unique objects
    # always sample the smallest from the bin
    extracted_names = sorted([min(b, key=lambda s: int(s)) for b in new_bins])

    with open("dataset/dataset_gen_logs/filtered_id_list.json", 'w') as f:
        json.dump(extracted_names, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="/root/data",
                        help='dataset root.')
    parser.add_argument('--error_log', type=str,
                        default="dataset/dataset_gen_logs/3view_error.txt",
                        help='3 view rendering error log.')
    parser.add_argument('--similarity_threshold', type=float, default=0.1,
                        help="grouping threhold for jaccard distance")
    args = parser.parse_args()

    main(args)
